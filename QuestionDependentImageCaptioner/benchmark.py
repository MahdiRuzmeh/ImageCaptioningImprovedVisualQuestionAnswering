"""Benchmark yek checkpoint roye test split — architecture comparison.

Finglish:
    Checkpoint (har model_class) ro load mikone, test split (``test_row_range``)
    ro ba beam search decode mikone va ``test_metrics.json`` minevise:
    BLEU-1..4, CIDEr-D, ROUGE-L, AnswerHit (+ per question type), params, sorat.
    Model hyperparam ha az ``ckpt['config']``; split/path ha az ``--config``.

    python benchmark.py --config configs/bench_v1.yaml \\
        --ckpt outputs/bench/v1/seed42/best.pt --name v1
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

from decoding import generate_split_captions
from metrics import answer_hit, compute_caption_metrics
from models import build_captioner
from train import (
    PROJECT_ROOT,
    Vocab,
    build_split_dataset,
    collate_batch,
    load_config,
    load_model_state,
    region_cache_dir_for_split,
    resolve_path_fields,
    set_seed,
    tok,
)

PATH_KEYS = (
    "train_captions_json",
    "val_captions_json",
    "train_images_dir",
    "val_images_dir",
    "train_region_cache_dir",
    "val_region_cache_dir",
    "test_region_cache_dir",
)

# Finglish: model cfg az checkpoint; in key ha (data/split/cache) az bench config
DATA_KEYS = PATH_KEYS + (
    "train_row_range",
    "val_row_range",
    "test_row_range",
    "train_image_filename_template",
    "val_image_filename_template",
    "cache_regions",
    "images_from_cache",
    "max_caption_len",
    "max_question_len",
    "batch_size",
    "num_workers",
    "image_size",
)


def vocab_from_itos(itos: List[str]) -> Vocab:
    v = Vocab.__new__(Vocab)
    v.itos = list(itos)
    v.stoi = {w: i for i, w in enumerate(v.itos)}
    return v


def score_block(
    hyps: List[List[str]], samples: List[Dict[str, Any]]
) -> Dict[str, float]:
    """Finglish: caption metric + AnswerHit roye yek zir-majmooe."""
    refs = [[tok(s["caption"])] for s in samples]
    out = compute_caption_metrics(hyps, refs)
    out.update(answer_hit(hyps, [tok(s["answer"]) for s in samples]))
    out["n"] = float(len(samples))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark QD captioner on test split")
    p.add_argument("--config", required=True, help="Bench YAML (splits/paths)")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--name", required=True, help="Architecture label (e.g. v1)")
    p.add_argument("--split", default="test", choices=("val", "test"))
    p.add_argument("--beam", type=int, default=3)
    p.add_argument("--out", default=None, help="Output JSON (default kenar ckpt)")
    args = p.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    bench_cfg = load_config(str(config_path))
    resolve_path_fields(bench_cfg, PATH_KEYS)
    set_seed(int(bench_cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_cfg: Dict[str, Any] = dict(state.get("config") or {})
    # Finglish: model hyperparam az ckpt; data/split az bench config
    cfg = {**model_cfg, **{k: bench_cfg[k] for k in DATA_KEYS if k in bench_cfg}}

    vocab = vocab_from_itos(state["vocab"])
    q_vocab = vocab_from_itos(state["q_vocab"])
    model = build_captioner(cfg, vocab, q_vocab)
    load_model_state(model, state["model"])
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    ds = build_split_dataset(cfg, args.split, vocab, q_vocab)
    loader = DataLoader(
        ds,
        batch_size=int(cfg.get("batch_size", 16)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )
    cache_dir = region_cache_dir_for_split(cfg, args.split)
    hyps, secs = generate_split_captions(
        model,
        loader,
        vocab.itos,
        device,
        int(cfg["max_caption_len"]),
        beam_size=args.beam,
        region_cache_dir=cache_dir,
    )

    samples = ds.samples
    overall = score_block(hyps, samples)
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        groups[s["rule"] or "unknown"].append(i)
    per_rule = {
        rule: score_block([hyps[i] for i in idx], [samples[i] for i in idx])
        for rule, idx in sorted(groups.items())
    }

    preds = [
        {
            "image_id": s["image_id"],
            "question": s["question"],
            "answer": s["answer"],
            "rule": s["rule"],
            "gt": s["caption"],
            "pred": " ".join(h),
        }
        for s, h in zip(samples, hyps)
    ]
    result = {
        "name": args.name,
        "model_class": cfg.get("model_class", "QuestionDependentImageCaptioner"),
        "seed": model_cfg.get("seed"),
        "ckpt": str(ckpt_path),
        "best_epoch": state.get("best_epoch", state.get("epoch")),
        "select_metric": state.get("select_metric", "loss"),
        "split": args.split,
        "beam": args.beam,
        "trainable_params": n_params,
        "decode_secs": secs,
        "items_per_sec": len(samples) / max(secs, 1e-9),
        "overall": overall,
        "per_rule": per_rule,
        "predictions": preds,
    }
    out = Path(args.out) if args.out else ckpt_path.parent / f"{args.split}_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")

    print(f"{args.name} ({result['model_class']}) seed={result['seed']} "
          f"best_epoch={result['best_epoch']} params={n_params:,}")
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in overall.items() if k != "n"))
    for rule, sc in per_rule.items():
        print(f"  [{rule}] n={int(sc['n'])} CIDEr={sc['CIDEr']:.4f} "
              f"AnswerHit={sc['AnswerHit']:.4f}")
    for row in preds[:5]:
        print(f"  Q: {row['question']}\n    pred: {row['pred']}\n    gt:   {row['gt']}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
