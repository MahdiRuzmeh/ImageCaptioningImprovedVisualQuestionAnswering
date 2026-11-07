"""Kaggle runner: resume LLM caption generation with Ollama, stop cleanly before the time limit.

Kaggle notebook (GPU T4 x2, Internet ON), one cell, then
"Save Version -> Save & Run All (Commit)":

    !git clone https://github.com/MahdiRuzmeh/ImageCaptioningImprovedVisualQuestionAnswering.git /tmp/repo
    !git -C /tmp/repo checkout feat/kaggle-runner
    !python -u /tmp/repo/QuestionDependentCaptionGenerator/kaggle/kaggle_runner.py

All settings live in kaggle_config.json (next to this file). Options:

    kaggle_runner.py [--config OVERRIDE.json] [--dry-run]

  --config   extra JSON merged over kaggle_config.json (only the keys you
             want to change), e.g. a file you write in a notebook cell
  --dry-run  build the layout, resolve every input path, print the
             generate.py command, then stop (no Ollama, no generation)

(Pasting this whole file into a cell also works; it then clones by itself.)

The run:
  1. copies the generator code from the checkout and builds the normal folder
     layout under <work_dir>/src/QuestionDependentCaptionGenerator, so
     generate.py resumes exactly as on the PC (VQA JSONs are passed with
     --questions / --annotations straight from /kaggle/input);
  2. installs Ollama, starts it, pulls the model;
  3. runs generate.py --llm and sends Ctrl+C (SIGINT) after run.hours, which
     makes generate.py write its checkpoint and exit;
  4. leaves only outputs/ (and ollama.log) in <work_dir> -> notebook output.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

GEN_NAME = "QuestionDependentCaptionGenerator"
CONFIG_NAME = "kaggle_config.json"
# Only needed when this file was pasted into a cell (no checkout to read the
# config from yet). Keep in sync with kaggle_config.json -> repo.
PASTE_REPO = {
    "url": "https://github.com/MahdiRuzmeh/ImageCaptioningImprovedVisualQuestionAnswering.git",
    "branch": "feat/kaggle-runner",
    "clone_dir": "/tmp/repo",
}

T0 = time.time()


def sh(cmd: str) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_input(value: str, search_root: str) -> Path:
    """Absolute path -> use as is. Bare file name -> search under search_root."""
    p = Path(value)
    if p.is_absolute():
        if not p.is_file():
            raise FileNotFoundError(f"configured path does not exist: {p}")
        return p
    hits = sorted(glob.glob(f"{search_root}/**/{value}", recursive=True))
    if not hits:
        raise FileNotFoundError(
            f"{value} not found under {search_root}. Attach the dataset that "
            f"has it, or set its absolute path in {CONFIG_NAME}."
        )
    if len(hits) > 1:
        print(f"  note: {value} found {len(hits)}x, using {hits[0]}", flush=True)
    return Path(hits[0])


def count_needs_llm(captions_file: Path) -> int:
    with open(captions_file, encoding="utf-8") as f:
        rows = json.load(f)["annotations"]
    return sum(1 for r in rows if r.get("rule") == "needs_llm")


def find_repo() -> Path:
    """Checkout this file runs from; else clone + checkout (paste mode)."""
    if "__file__" in globals():
        here = Path(__file__).resolve()
        if len(here.parents) > 2 and (here.parents[2] / GEN_NAME).is_dir():
            return here.parents[2]
    repo = Path(PASTE_REPO["clone_dir"])
    if not repo.exists():
        sh(f"git clone {PASTE_REPO['url']} {repo}")
    sh(f"git -C {repo} checkout {PASTE_REPO['branch']}")
    return repo


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=str, default=None,
                    help="JSON merged over kaggle_config.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve paths + print command, then stop")
    # parse_known_args: ignore Jupyter's own -f argument in paste mode
    return ap.parse_known_args()[0]


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------ config
    repo = find_repo()
    subprocess.run(["git", "-C", str(repo), "log", "--oneline", "-1"])
    cfg_path = repo / GEN_NAME / "kaggle" / CONFIG_NAME
    cfg = load_json(cfg_path)
    if args.config:
        cfg = deep_merge(cfg, load_json(Path(args.config)))
        print(f"Config: {cfg_path} + override {args.config}", flush=True)
    else:
        print(f"Config: {cfg_path}", flush=True)
    print(json.dumps(cfg, indent=2), flush=True)

    inp, run, gen_cfg, oll = cfg["inputs"], cfg["run"], cfg["generate"], cfg["ollama"]
    split = gen_cfg["split"]
    captions_name = f"vqa_v2_question_dependent_captions_{split}2014.json"
    work = Path(cfg["paths"]["work_dir"])
    src = work / "src"
    gen = src / GEN_NAME
    out = gen / "outputs"

    # ------------------------------------------------------------ 1. layout
    shutil.copytree(repo / GEN_NAME, gen, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("outputs", "__pycache__",
                                                  ".codegraphy"))

    questions = resolve_input(inp["questions"], inp["search_root"])
    annotations = resolve_input(inp["annotations"], inp["search_root"])
    print(f"Questions:   {questions}\nAnnotations: {annotations}", flush=True)

    # outputs: explicit folders from config, else every folder that holds the
    # captions JSON (your upload, previous notebook versions)
    if inp["outputs_dirs"]:
        candidates = [Path(d) for d in inp["outputs_dirs"]]
        missing = [d for d in candidates if not (d / captions_name).is_file()]
        if missing:
            raise FileNotFoundError(f"{captions_name} missing in: {missing}")
    else:
        candidates = [Path(p).parent for p in sorted(glob.glob(
            f"{inp['search_root']}/**/{captions_name}", recursive=True))]
        if not candidates:
            raise FileNotFoundError(
                f"{captions_name} not found under {inp['search_root']}")
    scored = sorted((count_needs_llm(d / captions_name), str(d))
                    for d in candidates)
    for n, d in scored:
        print(f"  {n:>7} needs_llm  {d}", flush=True)
    latest = Path(scored[0][1])                   # most progress wins
    print(f"Resuming from: {latest}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    for f in latest.glob("vqa_v2_*"):              # generator files only
        shutil.copy2(f, out / f.name)

    host = oll["host"]
    cmd: List[str] = [
        sys.executable, "-u", "generate.py", "--split", split, "--llm",
        "--questions", str(questions), "--annotations", str(annotations),
        "--model", gen_cfg["model"],
        "--batch-size", str(gen_cfg["batch_size"]),
        "--llm-parallel", str(gen_cfg["llm_parallel"]),
        "--checkpoint-every", str(gen_cfg["checkpoint_every"]),
        "--ollama-host", host,
        *[str(a) for a in gen_cfg["extra_args"]],
    ]
    print("$ (cd " + str(gen) + ") " + " ".join(cmd), flush=True)

    if args.dry_run:
        print("--dry-run: stopping before Ollama / generation.", flush=True)
        return

    # ------------------------------------------------------------ 2. ollama
    if shutil.which("ollama") is None:
        sh("apt-get -qq update && apt-get -qq install -y zstd > /dev/null")
        sh("curl -fsSL https://ollama.com/install.sh | sh")

    num_parallel = oll["num_parallel"] or gen_cfg["llm_parallel"]
    env = dict(os.environ,
               OLLAMA_HOST=urllib.parse.urlparse(host).netloc,
               OLLAMA_NUM_PARALLEL=str(num_parallel),
               OLLAMA_KEEP_ALIVE=str(oll["keep_alive"]))
    if oll["sched_spread"]:
        env["OLLAMA_SCHED_SPREAD"] = "1"           # spread across both T4s
    ollama = subprocess.Popen(["ollama", "serve"], env=env,
                              stdout=open(cfg["paths"]["ollama_log"], "w"),
                              stderr=subprocess.STDOUT)
    for _ in range(60):
        try:
            urllib.request.urlopen(f"{host}/api/tags", timeout=2)
            break
        except Exception:
            time.sleep(2)
    subprocess.run(["ollama", "pull", gen_cfg["model"]], env=env, check=True)

    # ------------------------------------------------------------ 3. generate
    proc = subprocess.Popen(cmd, cwd=gen)
    deadline = time.time() + float(run["hours"]) * 3600
    while proc.poll() is None and time.time() < deadline:
        time.sleep(30)

    if proc.poll() is None:
        print(f"\n{run['hours']}h reached -> sending Ctrl+C, waiting for "
              f"checkpoint...", flush=True)
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=float(run["shutdown_grace_min"]) * 60)
        except subprocess.TimeoutExpired:
            print("generate.py did not exit in time; killing (last periodic "
                  "checkpoint is still on disk).", flush=True)
            proc.kill()
            proc.wait()
    print(f"generate.py exit code: {proc.returncode}", flush=True)

    # ------------------------------------------------------------ 4. tidy
    ollama.terminate()
    for p in gen.iterdir():                       # keep only outputs/ as result
        if p.name != "outputs":
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink()
    left = count_needs_llm(out / captions_name)
    print(f"Remaining needs_llm: {left}", flush=True)
    print(f"Total wall time: {(time.time() - T0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
