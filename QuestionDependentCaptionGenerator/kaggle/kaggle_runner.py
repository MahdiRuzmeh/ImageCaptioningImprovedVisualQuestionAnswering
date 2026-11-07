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
  2. installs Ollama, starts ONE `ollama serve` PER GPU (each pinned with
     CUDA_VISIBLE_DEVICES, own port), pulls + loads the model on each, and
     puts a load-balancing proxy on ollama.host so requests are split
     between the GPUs (generate.py itself is unchanged);
  3. runs generate.py --llm and sends Ctrl+C (SIGINT) after run.hours, which
     makes generate.py write its checkpoint and exit;
  4. leaves only outputs/ (and the ollama_gpu*.log files) in <work_dir>
     -> notebook output.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
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


def require_file(value: str, key: str) -> Path:
    """Config paths are used as-is: absolute, and the file must exist."""
    p = Path(value)
    if not p.is_absolute():
        raise ValueError(f"{key} must be an absolute path, got {value!r}")
    if not p.is_file():
        raise FileNotFoundError(
            f"{key}: {p} does not exist. Check the path with "
            f"`!ls {p.parent}` and fix it in {CONFIG_NAME} or --config.")
    return p


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


# ---------------------------------------------------------------- multi-GPU
def detect_gpus(setting: Any) -> List[str]:
    """"auto" -> one entry per GPU from nvidia-smi; a list -> as given."""
    if setting != "auto":
        return [str(g) for g in setting]
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                             text=True, check=True).stdout
        n = sum(1 for line in out.splitlines() if line.startswith("GPU "))
    except (OSError, subprocess.CalledProcessError):
        n = 0
    return [str(i) for i in range(n)] or [""]   # "" = no GPU pinning


def wait_ready(url: str, timeout_s: float = 120) -> None:
    end = time.time() + timeout_s
    while time.time() < end:
        try:
            urllib.request.urlopen(f"{url}/api/tags", timeout=2)
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"Ollama at {url} did not start in {timeout_s}s")


class BalancingProxy:
    """Tiny HTTP proxy: each request goes to the backend with the fewest
    requests in flight (ties -> round-robin), so N GPUs share the load.

    generate.py keeps talking to one --ollama-host; it never knows about the
    backends. Ollama calls here are non-streaming, so buffering is fine.
    """

    def __init__(self, listen: str, backends: List[str],
                 timeout_s: float = 900) -> None:
        import http.server
        import itertools
        import threading

        self.backends = backends
        self.active = [0] * len(backends)
        self.served = [0] * len(backends)
        self.failed = [0] * len(backends)
        self.lock = threading.Lock()
        self._rr = itertools.count()
        proxy = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a: Any) -> None:   # keep the log quiet
                pass

            def _forward(self) -> None:
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n else None
                i = proxy._pick()
                req = urllib.request.Request(
                    proxy.backends[i] + self.path, data=body,
                    method=self.command,
                    headers={"Content-Type": self.headers.get(
                        "Content-Type", "application/json")})
                status, data, ctype = 502, b"proxy error", "text/plain"
                try:
                    with urllib.request.urlopen(req, timeout=timeout_s) as r:
                        status, data = r.status, r.read()
                        ctype = r.headers.get("Content-Type", "application/json")
                except urllib.error.HTTPError as e:
                    status, data = e.code, e.read()
                    ctype = e.headers.get("Content-Type", "text/plain")
                except Exception as e:                 # backend down / timeout
                    data = f"backend {proxy.backends[i]}: {e}".encode()
                finally:
                    proxy._release(i, ok=status < 500)
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST = do_DELETE = do_HEAD = _forward

        u = urllib.parse.urlparse(listen)
        self.server = http.server.ThreadingHTTPServer(
            (u.hostname or "127.0.0.1", u.port or 80), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)

    def _pick(self) -> int:
        with self.lock:
            low = min(self.active)
            idle = [i for i, a in enumerate(self.active) if a == low]
            i = idle[next(self._rr) % len(idle)]
            self.active[i] += 1
            return i

    def _release(self, i: int, ok: bool) -> None:
        with self.lock:
            self.active[i] -= 1
            self.served[i] += 1
            if not ok:
                self.failed[i] += 1

    def start(self) -> "BalancingProxy":
        self.thread.start()
        return self

    def stop(self) -> None:
        self.server.shutdown()

    def stats(self) -> str:
        return ", ".join(f"{b}: {s} requests ({f} failed)" for b, s, f in
                         zip(self.backends, self.served, self.failed))


def start_ollama_servers(oll: Dict[str, Any], llm_parallel: int, log_dir: Path):
    """One `ollama serve` per GPU on base_port, base_port+1, ...

    Returns (processes, backend URLs). Each server gets
    ceil(llm_parallel / n) parallel slots so the total matches llm_parallel.
    """
    gpus = detect_gpus(oll["gpus"])
    per_server = oll["num_parallel"] or -(-llm_parallel // len(gpus))
    procs, urls = [], []
    for k, gpu in enumerate(gpus):
        port = int(oll["base_port"]) + k
        env = dict(os.environ,
                   OLLAMA_HOST=f"127.0.0.1:{port}",
                   OLLAMA_NUM_PARALLEL=str(per_server),
                   OLLAMA_KEEP_ALIVE=str(oll["keep_alive"]))
        if gpu != "":
            env["CUDA_VISIBLE_DEVICES"] = gpu
        log = log_dir / f"ollama_gpu{gpu or 'X'}.log"
        procs.append(subprocess.Popen(["ollama", "serve"], env=env,
                                      stdout=open(log, "w"),
                                      stderr=subprocess.STDOUT))
        urls.append(f"http://127.0.0.1:{port}")
        print(f"Ollama #{k}: GPU {gpu or '-'} -> {urls[-1]} "
              f"(NUM_PARALLEL={per_server}, log {log})", flush=True)
    for u in urls:
        wait_ready(u)
    return procs, urls


def warm_up(urls: List[str], model: str) -> None:
    """Load the model on every server now, not during the first batches."""
    body = json.dumps({"model": model, "prompt": "", "keep_alive": -1}).encode()
    for u in urls:
        req = urllib.request.Request(f"{u}/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=600).read()
        print(f"  model loaded on {u}", flush=True)


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

    questions = require_file(inp["questions"], "inputs.questions")
    annotations = require_file(inp["annotations"], "inputs.annotations")
    resume_dir = Path(inp["outputs_dir"])
    resume_file = require_file(str(resume_dir / captions_name),
                               "inputs.outputs_dir")
    print(f"Questions:   {questions}", flush=True)
    print(f"Annotations: {annotations}", flush=True)
    # /kaggle/input is read-only: copy the generator files into the writable
    # outputs/ folder; generate.py resumes from (and writes to) that copy.
    print(f"Resuming from: {resume_dir} "
          f"({count_needs_llm(resume_file)} needs_llm)", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    for f in resume_dir.glob("vqa_v2_*"):              # generator files only
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

    # one server per GPU (models dir is shared, so one pull serves all)
    servers, backends = start_ollama_servers(
        oll, int(gen_cfg["llm_parallel"]), Path(cfg["paths"]["log_dir"]))
    pull_env = dict(os.environ,
                    OLLAMA_HOST=urllib.parse.urlparse(backends[0]).netloc)
    subprocess.run(["ollama", "pull", gen_cfg["model"]], env=pull_env,
                   check=True)
    warm_up(backends, gen_cfg["model"])
    # generate.py talks to `host`; the proxy splits requests over the GPUs
    proxy = BalancingProxy(host, backends).start()
    wait_ready(host)
    print(f"Proxy {host} -> {len(backends)} Ollama server(s)", flush=True)

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
    print(f"Load split: {proxy.stats()}", flush=True)
    proxy.stop()
    for s in servers:
        s.terminate()
    for p in gen.iterdir():                       # keep only outputs/ as result
        if p.name != "outputs":
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink()
    left = count_needs_llm(out / captions_name)
    print(f"Remaining needs_llm: {left}", flush=True)
    print(f"Total wall time: {(time.time() - T0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
