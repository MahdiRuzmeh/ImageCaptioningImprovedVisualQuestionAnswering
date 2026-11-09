"""Kaggle runner: find WRONG captions and regenerate them (2 GPUs, resumable).

Kaggle notebook (GPU T4 x2, Internet ON), one cell, then
"Save Version -> Save & Run All (Commit)":

    !git clone https://github.com/MahdiRuzmeh/ImageCaptioningImprovedVisualQuestionAnswering.git /tmp/repo
    !git -C /tmp/repo checkout feat/kaggle-runner
    !python -u /tmp/repo/QuestionDependentCaptionGenerator/kaggle/kaggle_judge_runner.py

All settings live in kaggle_judge_config.json (next to this file):

    kaggle_judge_runner.py [--config OVERRIDE.json] [--dry-run]

run.mode:
  "calibrate"  judge only the hand-labelled audit rows (inputs.audit) and print
               precision / recall of WRONG. A few minutes. Do this first.
  "full"       judge every caption of run.scope and regenerate the WRONG
               ones (caption_errors/judge_captions.py).

run.scope (full mode):
  "all"        every row
  "manual"     only rows whose caption_status is "Need to Manual validate"
               (~7.5k rows). A later "all" run with resume_dir skips them.

The run:
  1. checks every input path (and the captions row count), copies the
     previous session's progress file (inputs.resume_dir, optional) into the
     writable output dir;
  2. starts ONE `ollama serve` PER GPU behind a load-balancing proxy (same
     code as kaggle_runner.py), pulls + loads the model on each GPU; each
     server runs several requests at once (OLLAMA_NUM_PARALLEL);
  3. runs the job with a time limit of run.hours counted from when THIS
     runner started (setup included). The job stops starting new batches
     5 min before the limit and saves; at the limit it gets Ctrl+C, and
     run.shutdown_grace_min later it is killed;
  4. every session ends with a captions JSON in the output dir (rows judged so
     far carry caption_quality_verdict / verdict_reason, regenerated rows the
     new caption). Next session: attach this output, set inputs.resume_dir.
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
CONFIG_NAME = "kaggle_judge_config.json"
PROGRESS_SUFFIX = "_caption_check_progress.jsonl"   # = caption_errors.judge_captions
SELF_STOP_BEFORE_S = 300                            # job saves 5 min before run.hours
T0 = time.time()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None,
                    help="JSON merged over kaggle_judge_config.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve paths + print command, then stop")
    return ap.parse_known_args()[0]


def resolve_in_repo(value: str, key: str) -> Path:
    p = Path(value)
    return require_file(str(p if p.is_absolute() else GEN_DIR / p), key)


def main() -> None:
    args = parse_args()
    subprocess.run(["git", "-C", str(GEN_DIR.parent), "log", "--oneline", "-1"])
    cfg = load_json(HERE / CONFIG_NAME)
    if args.config:
        cfg = deep_merge(cfg, load_json(Path(args.config)))
    print(json.dumps(cfg, indent=2), flush=True)

    inp, run, jd, oll = cfg["inputs"], cfg["run"], cfg["judge"], cfg["ollama"]
    rg = cfg["regen"]
    mode = run["mode"]
    if mode not in ("calibrate", "full"):
        raise SystemExit(f"run.mode must be 'calibrate' or 'full', got {mode!r}")
    scope = run.get("scope", "all")
    if scope not in ("all", "manual"):
        raise SystemExit(f"run.scope must be 'all' or 'manual', got {scope!r}")
    out_dir = Path(cfg["paths"]["output_dir"])
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    deadline = T0 + float(run["hours"]) * 3600
    common = ["--model", jd["model"], "--ollama-host", oll["host"],
              "--llm-parallel", str(jd["llm_parallel"]),
              "--batch-size", str(jd["batch_size"]),
              "--num-ctx", str(jd["num_ctx"]),
              "--timeout", str(jd["timeout_s"]), "--retries", str(jd["retries"])]

    if mode == "calibrate":
        audit = resolve_in_repo(inp["audit"], "inputs.audit")
        cmd: List[str] = [
            sys.executable, "-u", "audit/audit_caption_errors.py", str(audit),
            "--output", str(out_dir / (audit.stem + "_scored.json")), *common]
        print(f"Mode: calibrate on {audit}", flush=True)
    else:
        captions = require_file(inp["captions"], "inputs.captions")
        with open(captions, encoding="utf-8") as f:
            n_rows = len(json.load(f)["annotations"])
        print(f"Captions: {captions} ({n_rows} rows), scope: {scope}", flush=True)
        want = inp.get("expected_rows")
        if want and int(want) != n_rows:
            raise SystemExit(
                f"inputs.captions has {n_rows} rows, inputs.expected_rows is "
                f"{want}. Wrong captions file? Fix the path or expected_rows.")
        progress_name = captions.stem + PROGRESS_SUFFIX
        if inp.get("resume_dir"):
            prev = require_file(str(Path(inp["resume_dir"]) / progress_name),
                                "inputs.resume_dir")
            n = sum(1 for _ in open(prev, encoding="utf-8")) - 1   # minus header
            if not args.dry_run:
                shutil.copy2(prev, out_dir / progress_name)
            print(f"Resuming from {prev} ({n} progress lines)", flush=True)
        else:
            print("No inputs.resume_dir: starting from row 0.", flush=True)
        cmd = [
            sys.executable, "-u", "-m", "caption_errors.judge_captions",
            "--input", str(captions), "--output-dir", str(out_dir), *common,
            "--scope", scope,
            "--regen-batch-size", str(rg["batch_size"]),
            "--regen-rounds", str(rg["rounds"]),
            "--regen-temperature", str(rg["temperature"]),
            *([] if rg["single_fallback"] else ["--no-regen-single-fallback"]),
            "--deadline-unix", str(deadline - SELF_STOP_BEFORE_S),
            *[str(a) for a in jd["extra_args"]],
        ]
    print("$ (cd " + str(GEN_DIR) + ") " + " ".join(cmd), flush=True)
    print(f"Time limit: run.hours={run['hours']} from runner start "
          f"({(deadline - time.time()) / 3600:.2f} h left now).", flush=True)
    if args.dry_run:
        print("--dry-run: stopping before Ollama.", flush=True)
        return

    # ------------------------------------------------------------ ollama x GPUs
    if shutil.which("ollama") is None:
        sh("apt-get -qq update && apt-get -qq install -y zstd > /dev/null")
        sh("curl -fsSL https://ollama.com/install.sh | sh")
    servers, backends = start_ollama_servers(
        oll, int(jd["llm_parallel"]), Path(cfg["paths"]["log_dir"]))
    pull_env = dict(os.environ,
                    OLLAMA_HOST=urllib.parse.urlparse(backends[0]).netloc)
    subprocess.run(["ollama", "pull", jd["model"]], env=pull_env, check=True)
    warm_up(backends, jd["model"])
    proxy = BalancingProxy(oll["host"], backends).start()
    wait_ready(oll["host"])
    print(f"Proxy {oll['host']} -> {len(backends)} Ollama server(s); setup took "
          f"{(time.time() - T0) / 60:.1f} min", flush=True)

    # ------------------------------------------------------------ job
    proc = subprocess.Popen(cmd, cwd=GEN_DIR)
    while proc.poll() is None and time.time() < deadline:
        time.sleep(min(30, max(1, deadline - time.time())))
    if proc.poll() is None:
        print(f"\n{run['hours']} h since start reached -> Ctrl+C (progress is "
              f"saved per batch)...", flush=True)
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=float(run["shutdown_grace_min"]) * 60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    code = proc.returncode
    print(f"Job exit code: {code}", flush=True)

    print(f"Load split: {proxy.stats()}", flush=True)
    proxy.stop()
    for s in servers:
        s.terminate()
    if mode == "calibrate":
        print("Calibration done: see the scores above and the *_scored.json "
              "file in the output dir.", flush=True)
    elif code == 0:
        print(f"DONE: every row of scope '{scope}' judged; the captions JSON is "
              "in the output dir (info.caption_error_fix.progress.complete = "
              "true). " + ("To judge the other rows too, run again with "
                           'run.scope "all" and inputs.resume_dir = this output.'
                           if scope != "all" else ""), flush=True)
    else:
        print("NOT finished (the captions JSON in the output dir holds "
              "everything done so far). Next session: attach this version's "
              f"output as input and set inputs.resume_dir to the folder holding "
              f"{progress_name}.", flush=True)
    print(f"Total wall time: {(time.time() - T0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
