"""Evaluate trained runs (best.pt) on a split of ANOTHER work dir — cross-condition test.

Finglish: run ha az ``--runs-config`` (answer vocab = train.json-e hamoon work dir),
data az ``--data-config`` work dir / ``--split``. Masalan model haye train shode
roye image haye unseen (small) → test roye image haye train-e captioner (seen):

    python eval_runs.py --runs-config configs/small.yaml --data-config configs/seen.yaml --split test
    python eval_runs.py --runs-config configs/seen.yaml --data-config configs/small.yaml --split test

Natije: ``<data work>/evals/<runs work name>/<run>_<split>.json`` + jadval-e mean ± std.
Breakdown: answer type, va ``qd_pair`` (hamin (image, soal) dar train-e captioner bude).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List

import torch

from common import load_config, read_json, write_json
from train import VQAFeatures, Vocabs, build_model, evaluate


def breakdown(preds: List[dict], recs: List[dict]) -> Dict[str, float]:
    by_qid = {r["question_id"]: r for r in recs}
    groups: Dict[str, List[float]] = defaultdict(list)
    for p in preds:
        r = by_qid[p["question_id"]]
        if "qd_pair" in r:
            key = "qd_pair" if r["qd_pair"] else "other_q"
            groups[key].append(p["score"])
            groups[f"{key}_{r['answer_type']}"].append(p["score"])
    return {f"acc_{k}": 100 * sum(v) / len(v) for k, v in sorted(groups.items())} | {
        f"n_{k}": len(v) for k, v in sorted(groups.items())}


def fmt(vals: List[float]) -> str:
    return f"{mean(vals):.2f} ± {stdev(vals):.2f}" if len(vals) > 1 else f"{vals[0]:.2f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-config", required=True)
    p.add_argument("--data-config", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--runs", default="", help="comma-separated run names (default: all generated/nocap runs)")
    args = p.parse_args()
    rcfg, dcfg = load_config(args.runs_config), load_config(args.data_config)
    rwork, dwork = Path(rcfg["work_dir"]), Path(dcfg["work_dir"])
    device = torch.device("cuda")

    cap_ckpt = torch.load(rcfg["captioner_ckpt"], map_location="cpu", weights_only=False)
    vocabs = Vocabs(read_json(rwork / "split" / "train.json"), int(rcfg["answer_min_count"]), cap_ckpt)
    names = [n for n in args.runs.split(",") if n] or sorted(
        d.name for d in (rwork / "runs").iterdir()
        if (d / "results.json").exists() and (d / "best.pt").exists() and "_human" not in d.name)
    datasets: Dict[bool, VQAFeatures] = {}
    out_dir = dwork / "evals" / rwork.name
    table: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for name in names:
        res = read_json(rwork / "runs" / name / "results.json")
        model_name, use_caption = res["model"], res["use_caption"]
        want_c = model_name == "C" and use_caption
        if want_c not in datasets:
            datasets[want_c] = VQAFeatures(args.split, dwork, vocabs, want_c)
        ds = datasets[want_c]
        ds.perm = torch.arange(len(ds))
        model = build_model(model_name, rcfg, vocabs, cap_ckpt, use_caption).to(device)
        model.load_state_dict(torch.load(rwork / "runs" / name / "best.pt")["model"])
        ev = evaluate(model, ds, vocabs, device, keep_preds=True)
        ev.update(breakdown(ev.pop("predictions"), ds.recs))
        out = {"run": name, "runs_work_dir": str(rwork), "data_work_dir": str(dwork),
               "split": args.split, "test": ev}
        if use_caption:
            ds.shuffled_captions(seed=123)
            sh = evaluate(model, ds, vocabs, device, keep_preds=True)
            sh.update(breakdown(sh.pop("predictions"), ds.recs))
            out["test_shuffled_caption"] = sh
            ds.perm = torch.arange(len(ds))
        write_json(out_dir / f"{name}_{args.split}.json", out)
        row = f"{model_name}{'' if use_caption else '_nocap'}"
        for k, v in ev.items():
            if k.startswith("acc") or k in ("pointer_rate", "gate_mean"):
                table[row][k].append(v)
        if use_caption:
            table[row]["shuffled"].append(out["test_shuffled_caption"]["acc"])
        print(f"{name}: acc {ev['acc']:.2f}"
              + (f" | qd_pair {ev['acc_qd_pair']:.2f} other_q {ev['acc_other_q']:.2f}" if "acc_qd_pair" in ev else "")
              + (f" | shuffled {out['test_shuffled_caption']['acc']:.2f}" if use_caption else ""))

    cols = ["acc", "acc_yes/no", "acc_number", "acc_other", "acc_qd_pair", "acc_other_q", "shuffled"]
    print(f"\n{rwork.name} runs on {dwork.name}/{args.split}")
    print("| Run | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for row, d in sorted(table.items()):
        print(f"| {row} | " + " | ".join(fmt(d[c]) if d.get(c) else "–" for c in cols) + " |")


if __name__ == "__main__":
    main()
