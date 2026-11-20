"""Ensemble of trained VQA models: average their answer scores, report VQA accuracy on dev and test.

Finglish: har run (folder ba best.pt + results.json) → score-e sigmoid baraye har javab (N, K), ye bar hesab mishe va
{dev,test}_scores.npy kenare best.pt save mishe (dafe-ye ba'd sari'). Model ha mitoonan vocab/javab-e motafavet
dashte bashan → score ha roye list-e moshtarak-e javab ha map mishan. Combination ha:
  - har model tanha, miangin-e seed ha (baseline / caption), hame ba ham;
  - wazn-e caption vs baseline (w = 0 … 1) → w roye DEV entekhab mishe, TEST faghat gozaresh mishe.

    python ensemble_vqa.py --config configs/blip_strong.yaml \
        --captions-json vqa_strong_crossfit/captions/captions_crossfit.json \
        --runs vqa_strong_baseline/runs/nocap_s0 vqa_strong_baseline/runs/nocap_s1 \
               vqa_strong_crossfit/runs/blip_s0 vqa_strong_crossfit/runs/blip_s1 --out vqa_ensemble.md
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from common import load_config, read_json, tok, vqa_score, write_json
from models_vqa import MCANVQA, UpDnVQA
from train_vqa import Split, load_captions, predict_scores


class CkptVocab:
    """Finglish: vocab va list-e javab az checkpoint (dagighan hamoon-e training)."""

    def __init__(self, itos: List[str], answers: List[str]) -> None:
        self.itos, self.answers = itos, answers
        self.stoi = {w: i for i, w in enumerate(itos)}
        self.a_stoi = {a: i for i, a in enumerate(answers)}

    def encode(self, text: str, max_len: int) -> List[int]:
        return [self.stoi.get(w, 1) for w in tok(text)[:max_len]]


def run_scores(run: Path, cfg: dict, blip, device) -> Dict[str, object]:
    res = read_json(run / "results.json")
    source = res["captions"]
    todo = [s for s in ("dev", "test") if not (run / f"{s}_scores.npy").exists()]
    # Finglish: score ha + answers.json hast → best.pt lazem nist (run haye jadid-e Kaggle)
    if not todo and (run / "answers.json").exists():
        answers = read_json(run / "answers.json")
    else:
        state = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
        vocab = CkptVocab(state["vocab"], state["answers"])
        answers = vocab.answers
    # Finglish: esm = <experiment>/<run> (blip_s0 dar chand experiment hast)
    out = {"name": f"{run.parent.parent.name}/{res['name']}", "captions": source, "answers": answers}
    if todo:
        use_caption = source != "none"
        if cfg.get("model", "updn") == "mcan":
            model = MCANVQA(len(vocab.itos), len(vocab.answers), int(cfg["word_dim"]), int(cfg["hidden_dim"]),
                            int(cfg.get("num_heads", 8)), int(cfg.get("ff_dim", 2048)), int(cfg.get("num_layers", 6)),
                            int(cfg["region_dim"]), float(cfg["dropout"]), use_caption)
        else:
            model = UpDnVQA(len(vocab.itos), len(vocab.answers), int(cfg["word_dim"]), int(cfg["hidden_dim"]),
                            int(cfg["region_dim"]), float(cfg["dropout"]), use_caption)
        model.load_state_dict(state["model"])
        model.to(device)
        for s in todo:
            ds = Split(s, cfg, vocab, source, blip)
            np.save(run / f"{s}_scores.npy", predict_scores(model, ds, len(vocab.answers), device))
            print(f"  {res['name']}: {s} scores computed ({len(ds)} questions)", flush=True)
        del model
        torch.cuda.empty_cache()
    for s in ("dev", "test"):
        out[s] = np.load(run / f"{s}_scores.npy").astype(np.float32)
    return out


def evaluate(scores: np.ndarray, answers: List[str], recs: List[dict]) -> Dict[str, float]:
    pred = scores.argmax(1)
    by = defaultdict(list)
    for r, a in zip(recs, pred):
        sc = vqa_score(answers[a], r["answers"])
        by["overall"].append(sc)
        by[r["answer_type"]].append(sc)
    return {k: 100 * sum(v) / len(v) for k, v in by.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/blip_strong.yaml")
    p.add_argument("--captions-json", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", default="vqa_ensemble.md")
    args = p.parse_args()
    cfg = load_config(args.config)
    cfg["captions_json"] = str(Path(args.captions_json).resolve())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blip = load_captions(cfg)
    work = Path(cfg["work_dir"])
    recs = {s: read_json(work / "split" / f"{s}.json") for s in ("dev", "test")}

    runs = [run_scores(Path(r), cfg, blip, device) for r in args.runs]
    # Finglish: list-e moshtarak-e javab ha (ejtema) → score-e har model roye oon map mishe
    answers = sorted({a for r in runs for a in r["answers"]})
    idx = {a: i for i, a in enumerate(answers)}
    for r in runs:
        cols = np.array([idx[a] for a in r["answers"]])
        for s in ("dev", "test"):
            full = np.zeros((r[s].shape[0], len(answers)), dtype=np.float32)
            full[:, cols] = r[s]
            r[s] = full
    print(f"{len(runs)} models, {len(answers)} answers in the common list")

    def mean(group, split):
        return np.mean([r[split] for r in group], axis=0)

    base = [r for r in runs if r["captions"] == "none"]
    cap = [r for r in runs if r["captions"] != "none"]
    combos = [(f"single: {r['name']}", lambda s, r=r: r[s]) for r in runs]
    if len(base) > 1:
        combos.append((f"baseline seeds averaged ({len(base)})", lambda s: mean(base, s)))
    if len(cap) > 1:
        combos.append((f"caption-model seeds averaged ({len(cap)})", lambda s: mean(cap, s)))
    if base and cap:
        combos.append((f"all {len(runs)} models, equal weight", lambda s: mean(runs, s)))
        # Finglish: wazn-e caption model ha (w) roye DEV
        grid = [round(0.1 * i, 1) for i in range(11)]
        dev_acc = {w: evaluate(w * mean(cap, "dev") + (1 - w) * mean(base, "dev"), answers, recs["dev"])["overall"]
                   for w in grid}
        w_best = max(grid, key=lambda w: dev_acc[w])
        print("dev accuracy by caption weight w: " + ", ".join(f"{w}: {dev_acc[w]:.2f}" for w in grid))
        combos.append((f"weighted: {w_best} × caption models + {1 - w_best:.1f} × baselines (w chosen on dev)",
                       lambda s: w_best * mean(cap, s) + (1 - w_best) * mean(base, s)))

    rows, result = [], {"answers": len(answers), "combos": []}
    for name, f in combos:
        d = evaluate(f("dev"), answers, recs["dev"])
        t = evaluate(f("test"), answers, recs["test"])
        result["combos"].append({"name": name, "dev": d, "test": t})
        rows.append(f"| {name} | {d['overall']:.2f} | **{t['overall']:.2f}** | {t.get('yes/no', 0):.2f} | "
                    f"{t.get('number', 0):.2f} | {t.get('other', 0):.2f} |")
        print(f"{name:75s} dev {d['overall']:.2f}  test {t['overall']:.2f}", flush=True)
    if base and cap:
        result["dev_by_caption_weight"] = dev_acc
    md = ["# VQA ensemble (test = 10,001 val2014 questions)", "",
          "| Combination | Dev | **Test** | Yes/no | Number | Other |", "|---|---:|---:|---:|---:|---:|", *rows]
    Path(args.out).write_text("\n".join(md) + "\n", encoding="utf-8")
    write_json(Path(args.out).with_suffix(".json"), result)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
