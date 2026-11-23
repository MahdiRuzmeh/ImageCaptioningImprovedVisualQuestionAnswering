"""EvalAI result file for VQA v2 test2015 (test-dev / test-standard) from one or more trained runs.

Finglish: har run (train_vqa.py ba split-e test2015) → test2015_scores.npy (fp16, N × K), answers.json,
test2015_question_ids.json. Ba chand run (e.g. 2 seed) score ha miangin mishan (chunk be chunk, memmap → RAM-e kam).
Khorooji = [{"question_id": int, "answer": str}, ...] baraye hame 447,793 soal — hamin file ro roye EvalAI upload kon
(phase-e test-dev ya test-standard).

    python scripts/test2015_results.py --runs /kaggle/working/vqa_final/runs/blip_s0 /kaggle/working/vqa_final/runs/blip_s1 \
        --questions <v2_OpenEnded_mscoco_test2015_questions.json> --out vqa_final_ensemble_test2015.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--questions", default=None, help="official test2015 questions JSON (check: every question answered)")
    p.add_argument("--chunk", type=int, default=20000)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    runs = [Path(r) for r in args.runs]
    answers = [json.loads((r / "answers.json").read_text(encoding="utf-8")) for r in runs]
    qids = [json.loads((r / "test2015_question_ids.json").read_text(encoding="utf-8")) for r in runs]
    for a, q in zip(answers[1:], qids[1:]):
        assert q == qids[0], "runs answer the test2015 questions in a different order"
    # Finglish: list-e moshtarak-e javab ha (age vocab-e javab ha fargh dasht)
    common = sorted(set().union(*answers))
    idx = {a: i for i, a in enumerate(common)}
    cols = [np.array([idx[a] for a in ans]) for ans in answers]
    S = [np.load(r / "test2015_scores.npy", mmap_mode="r") for r in runs]
    n = len(qids[0])
    for s in S:
        assert s.shape[0] == n
    out = []
    for c0 in range(0, n, args.chunk):
        c1 = min(n, c0 + args.chunk)
        acc = np.zeros((c1 - c0, len(common)), dtype=np.float32)
        for s, col in zip(S, cols):
            acc[:, col] += np.asarray(s[c0:c1], dtype=np.float32)
        for q, a in zip(qids[0][c0:c1], acc.argmax(1)):
            out.append({"question_id": int(q), "answer": common[int(a)]})
    if args.questions:
        want = {int(q["question_id"]) for q in json.loads(Path(args.questions).read_text(encoding="utf-8"))["questions"]}
        got = {r["question_id"] for r in out}
        assert got == want and len(out) == len(want), f"{len(want - got)} questions missing, {len(got - want)} extra"
        print(f"check: all {len(want)} test2015 questions answered exactly once")
    Path(args.out).write_text(json.dumps(out), encoding="utf-8")
    top = Counter(r["answer"] for r in out).most_common(8)
    print(f"{len(runs)} run(s) -> {len(out)} answers -> {args.out}")
    print("most common answers: " + ", ".join(f"{a} {100 * c / len(out):.1f}%" for a, c in top))


if __name__ == "__main__":
    main()
