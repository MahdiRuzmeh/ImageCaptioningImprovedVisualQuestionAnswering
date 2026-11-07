"""Kaggle runner: resume LLM caption generation with Ollama, stop cleanly before the time limit.

Paste this whole file into ONE Kaggle notebook cell (GPU T4 x2, Internet ON) and
use "Save Version -> Save & Run All (Commit)". The run:

  1. clones the code from GitHub and builds the normal folder layout under
     /kaggle/working/src (QuestionDependentCaptionGenerator/ + dataset/), so
     generate.py resumes exactly as on the PC;
  2. installs Ollama, starts it, pulls the same model;
  3. runs generate.py --llm and sends Ctrl+C (SIGINT) after RUN_HOURS, which
     makes generate.py write its checkpoint and exit;
  4. leaves only the outputs/ folder in /kaggle/working -> notebook output.

Inputs (Add Input in the notebook sidebar):
  - a Kaggle Dataset with the VQA question/annotation JSONs + the files of
    the outputs/ folder from the PC (first run; code comes from git);
  - for the next run: ALSO add this notebook's previous version output.
    The outputs/ copy with the fewest needs_llm rows wins automatically.
"""

from __future__ import annotations

import glob
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------- settings
REPO_URL = ("https://github.com/MahdiRuzmeh/"
            "ImageCaptioningImprovedVisualQuestionAnswering.git")
REPO_BRANCH = "master"     # code is cloned from here (public repo)
RUN_HOURS = 9.0            # send Ctrl+C after this many hours of generation
SHUTDOWN_GRACE_MIN = 30    # wait this long for the final checkpoint write
MODEL = "qwen2.5:3b-instruct-q4_K_M"   # keep identical to the PC run
BATCH_SIZE = 5             # PC run used 5
LLM_PARALLEL = 8           # concurrent Ollama requests
CHECKPOINT_EVERY = 50      # output JSON is ~150 MB; don't rewrite it every batch

WORK = Path("/kaggle/working")
SRC = WORK / "src"
GEN = SRC / "QuestionDependentCaptionGenerator"
DATA = SRC / "dataset"
OUT = GEN / "outputs"
CAPTIONS = "vqa_v2_question_dependent_captions_train2014.json"
QUESTIONS = "v2_OpenEnded_mscoco_train2014_questions.json"
ANNOTATIONS = "v2_mscoco_train2014_annotations.json"

T0 = time.time()


def sh(cmd: str) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def find_one(pattern: str) -> Path:
    hits = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    if not hits:
        raise FileNotFoundError(f"{pattern} not found under /kaggle/input")
    return Path(hits[0])


# ------------------------------------------------------------- 1. layout
REPO_DIR = Path("/tmp/repo")                  # outside /kaggle/working
if not REPO_DIR.exists():
    sh(f"git clone --depth 1 --branch {REPO_BRANCH} {REPO_URL} {REPO_DIR}")
shutil.copytree(REPO_DIR / "QuestionDependentCaptionGenerator", GEN,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("outputs", "__pycache__", ".codegraphy"))

DATA.mkdir(parents=True, exist_ok=True)
for name in (QUESTIONS, ANNOTATIONS):        # read-only is fine -> symlink
    link = DATA / name
    if not link.exists():
        link.symlink_to(find_one(name))

# every folder holding the captions JSON is a candidate (original upload,
# previous notebook versions); the one with the fewest needs_llm rows wins
candidates = [Path(p).parent for p in
              glob.glob(f"/kaggle/input/**/{CAPTIONS}", recursive=True)]
if not candidates:
    raise FileNotFoundError(f"{CAPTIONS} not found under /kaggle/input")

def remaining(d: Path) -> int:
    import json
    with open(d / CAPTIONS, encoding="utf-8") as f:
        rows = json.load(f)["annotations"]
    return sum(1 for r in rows if r.get("rule") == "needs_llm")


scored = sorted((remaining(d), str(d)) for d in candidates)
for n, d in scored:
    print(f"  {n:>7} needs_llm  {d}", flush=True)
latest = Path(scored[0][1])                   # most progress wins
print(f"Resuming from: {latest}", flush=True)
OUT.mkdir(parents=True, exist_ok=True)
for f in latest.glob("vqa_v2_*"):              # generator files only
    shutil.copy2(f, OUT / f.name)

# ------------------------------------------------------------- 2. ollama
if shutil.which("ollama") is None:
    sh("apt-get -qq update && apt-get -qq install -y zstd > /dev/null")
    sh("curl -fsSL https://ollama.com/install.sh | sh")

env = dict(os.environ,
           OLLAMA_NUM_PARALLEL=str(LLM_PARALLEL),
           OLLAMA_KEEP_ALIVE="-1",
           OLLAMA_SCHED_SPREAD="1")     # spread across both T4s
ollama = subprocess.Popen(["ollama", "serve"], env=env,
                          stdout=open(WORK / "ollama.log", "w"),
                          stderr=subprocess.STDOUT)
for _ in range(60):
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        break
    except Exception:
        time.sleep(2)
sh(f"ollama pull {MODEL}")

# ------------------------------------------------------------- 3. generate
cmd = [sys.executable, "-u", "generate.py", "--split", "train", "--llm",
       "--model", MODEL, "--batch-size", str(BATCH_SIZE),
       "--llm-parallel", str(LLM_PARALLEL),
       "--checkpoint-every", str(CHECKPOINT_EVERY)]
print("$ " + " ".join(cmd), flush=True)
proc = subprocess.Popen(cmd, cwd=GEN)

deadline = time.time() + RUN_HOURS * 3600
while proc.poll() is None and time.time() < deadline:
    time.sleep(30)

if proc.poll() is None:
    print(f"\n{RUN_HOURS}h reached -> sending Ctrl+C, waiting for checkpoint...",
          flush=True)
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=SHUTDOWN_GRACE_MIN * 60)
    except subprocess.TimeoutExpired:
        print("generate.py did not exit in time; killing (last periodic "
              "checkpoint is still on disk).", flush=True)
        proc.kill()
print(f"generate.py exit code: {proc.returncode}", flush=True)

# ------------------------------------------------------------- 4. tidy output
ollama.terminate()
shutil.rmtree(DATA, ignore_errors=True)       # don't publish VQA JSONs
for p in GEN.iterdir():                       # keep only outputs/ as result
    if p.name != "outputs":
        shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink()
sh(f"python -c \"import json,collections;"
   f"r=json.load(open('{OUT / CAPTIONS}'))['annotations'];"
   f"print(collections.Counter(x['rule'] for x in r).most_common(4))\"")
print(f"Total wall time: {(time.time() - T0) / 3600:.2f} h")
