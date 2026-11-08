"""Kaggle runner: re-check every captions row with the LLM question classifier (2 GPUs).

Kaggle notebook (GPU T4 x2, Internet ON), one cell, then
"Save Version -> Save & Run All (Commit)":

    !git clone https://github.com/MahdiRuzmeh/ImageCaptioningImprovedVisualQuestionAnswering.git /tmp/repo
    !git -C /tmp/repo checkout feat/kaggle-runner
    !python -u /tmp/repo/QuestionDependentCaptionGenerator/kaggle/kaggle_reclassify_runner.py

All settings live in kaggle_reclassify_config.json (next to this file):

    kaggle_reclassify_runner.py [--config OVERRIDE.json] [--dry-run]

The run:
  1. checks every input path from the config, copies the previous session's
     progress file (inputs.resume_dir, optional) into the writable output dir;
  2. starts ONE `ollama serve` PER GPU behind a load-balancing proxy (same
     code as kaggle_runner.py), pulls + loads the model on each GPU;
  3. runs reclassify_captions.py and sends Ctrl+C after run.hours; progress
     is saved after every batch, so the next session continues from it;
  4. when every row is classified, the output dir holds the new captions
     JSON (DIRECTLY_VISUAL only), the *_not_directly_visual_reclassified.json
     file and the updated classification result.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path
from typing import List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from kaggle_runner import (  # noqa: E402  (shared multi-GPU helpers)
    BalancingProxy, deep_merge, load_json, require_file, sh,
    start_ollama_servers, wait_ready, warm_up,
)

GEN_DIR = HERE.parent                      # .../QuestionDependentCaptionGenerator
CONFIG_NAME = "kaggle_reclassify_config.json"
PROGRESS_SUFFIX = "_reclassify_progress.jsonl"   # keep in sync with reclassify_captions.py
T0 = time.time()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None,
                    help="JSON merged over kaggle_reclassify_config.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve paths + print command, then stop")
    return ap.parse_known_args()[0]


def main() -> None:
    args = parse_args()
    subprocess.run(["git", "-C", str(GEN_DIR.parent), "log", "--oneline", "-1"])
    cfg = load_json(HERE / CONFIG_NAME)
    if args.config:
        cfg = deep_merge(cfg, load_json(Path(args.config)))
    print(json.dumps(cfg, indent=2), flush=True)

    inp, run, rc, oll = cfg["inputs"], cfg["run"], cfg["reclassify"], cfg["ollama"]
    captions = require_file(inp["captions"], "inputs.captions")
    clf_result = (require_file(inp["classification_result"],
                               "inputs.classification_result")
                  if inp.get("classification_result") else None)
    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Captions:              {captions}", flush=True)
    print(f"Classification result: {clf_result or '- (not updated)'}", flush=True)
    print(f"Output dir:            {out_dir}", flush=True)

    # Continue a previous session: its progress JSONL holds every label so far.
    progress_name = captions.stem + PROGRESS_SUFFIX
    if inp.get("resume_dir"):
        prev = require_file(str(Path(inp["resume_dir"]) / progress_name),
                            "inputs.resume_dir")
        shutil.copy2(prev, out_dir / progress_name)
        n = sum(1 for _ in open(prev, encoding="utf-8")) - 1   # minus header
        print(f"Resuming from {prev} ({n} rows already classified)", flush=True)
    else:
        print("No inputs.resume_dir: starting from row 0.", flush=True)

    cmd: List[str] = [
        sys.executable, "-u", "reclassify_captions.py",
        "--input", str(captions), "--output-dir", str(out_dir),
        "--model", rc["model"], "--ollama-host", oll["host"],
        "--llm-parallel", str(rc["llm_parallel"]),
        "--batch-size", str(rc["batch_size"]),
        "--timeout", str(rc["timeout_s"]), "--retries", str(rc["retries"]),
        *(["--classification-result", str(clf_result)] if clf_result else []),
        *[str(a) for a in rc["extra_args"]],
    ]
    print("$ (cd " + str(GEN_DIR) + ") " + " ".join(cmd), flush=True)
    if args.dry_run:
        print("--dry-run: stopping before Ollama / classification.", flush=True)
        return

    # ------------------------------------------------------------ ollama x GPUs
    if shutil.which("ollama") is None:
        sh("apt-get -qq update && apt-get -qq install -y zstd > /dev/null")
        sh("curl -fsSL https://ollama.com/install.sh | sh")
    servers, backends = start_ollama_servers(
        oll, int(rc["llm_parallel"]), Path(cfg["paths"]["log_dir"]))
    pull_env = dict(os.environ,
                    OLLAMA_HOST=urllib.parse.urlparse(backends[0]).netloc)
    subprocess.run(["ollama", "pull", rc["model"]], env=pull_env, check=True)
    warm_up(backends, rc["model"])
    proxy = BalancingProxy(oll["host"], backends).start()
    wait_ready(oll["host"])
    print(f"Proxy {oll['host']} -> {len(backends)} Ollama server(s)", flush=True)

    # ------------------------------------------------------------ classify
    proc = subprocess.Popen(cmd, cwd=GEN_DIR)
    deadline = time.time() + float(run["hours"]) * 3600
    while proc.poll() is None and time.time() < deadline:
        time.sleep(30)
    if proc.poll() is None:
        print(f"\n{run['hours']}h reached -> Ctrl+C (progress is saved per "
              f"batch)...", flush=True)
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=float(run["shutdown_grace_min"]) * 60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    code = proc.returncode
    print(f"reclassify_captions.py exit code: {code}", flush=True)

    print(f"Load split: {proxy.stats()}", flush=True)
    proxy.stop()
    for s in servers:
        s.terminate()
    if code == 0:
        print("DONE: all rows classified, final files are in the output dir.",
              flush=True)
    else:
        print("NOT finished: attach this version's output as input and set "
              f"inputs.resume_dir to the folder holding {progress_name}.",
              flush=True)
    print(f"Total wall time: {(time.time() - T0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
