"""Image-ablation diagnostic — model az tasvir estefade mikone ya faghat az soal?

Finglish:
    Checkpoint ro roye yek split decode mikone ba chand halat baraye region ha:
      normal  — region haye khod tasvir (= benchmark.py)
      shuffle — region haye yek tasvir *dige* az hamoon split (derangement sabet)
      mean    — hame region ha = miangin region haye split (bedoon etelaat tasvir)
      mask    — region ha kamelan mask (faghat MMTransformerCopyCaptioner; v1 NaN mide)
    Va yek baseline bedoon model:
      prior   — por-tekrar tarin javab train baraye (rule, 2 kalame aval soal)
    Age normal ≈ shuffle/mean → model "visually blind" hast.
    ImageDep = AnswerHit(normal) − AnswerHit(shuffle); Changed = sahm caption haye
    ke ba shuffle avaz mishan.

    python scripts/image_ablation.py --config configs/bench_transformer_copy.yaml \\
        --ckpt outputs/bench/transformer_copy/seed42/best.pt --split test \\
        --out outputs/ablation/transformer_copy_s42_test.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark import DATA_KEYS, PATH_KEYS, score_block, vocab_from_itos  # noqa: E402
from decoding import generate_split_captions  # noqa: E402
from metrics import NON_LEXICAL_ANSWERS  # noqa: E402
from models import build_captioner  # noqa: E402
from train import (  # noqa: E402
    PROJECT_ROOT,
    build_split_dataset,
    collate_batch,
    load_config,
    load_model_state,
    region_cache_dir_for_split,
    resolve_path_fields,
    set_seed,
    split_rows,
    tok,
)

MODES = ("normal", "shuffle", "mean", "mask")


def derangement(ids: List[int], seed: int) -> Dict[int, int]:
    """Finglish: har image_id → yek image_id *dige* (cyclic shift ba'd az permute)."""
    g = torch.Generator().manual_seed(seed)
    perm = [ids[i] for i in torch.randperm(len(ids), generator=g).tolist()]
    return {a: b for a, b in zip(perm, perm[1:] + perm[:1])}


def patch_regions(model: torch.nn.Module, mode: str, id_map: Dict[int, int],
                  mean_vec: torch.Tensor | None) -> None:
    """Finglish: region_encoder.forward ro avaz kon ta region ha ablate beshan."""
    enc = model.region_encoder
    if not hasattr(enc, "_orig_forward"):
        enc._orig_forward = enc.forward
    orig = enc._orig_forward

    def forward(images, image_ids=None, **kw):
        if mode == "shuffle":
            image_ids = torch.tensor(
                [id_map[int(i)] for i in image_ids.tolist()], device=image_ids.device
            )
        out = orig(images, image_ids=image_ids, **kw)
        if mode == "mean":
            out = mean_vec.to(out).expand_as(out).clone()
        elif mode == "mask":
            out = torch.zeros_like(out)
        return out

    enc.forward = forward


@torch.no_grad()
def split_mean_region(model, loader, device, cache_dir) -> torch.Tensor:
    """Finglish: miangin region vector (bad az roi_to_region) roye non-pad region haye split."""
    total, count = None, 0
    for batch in loader:
        r = model.region_encoder(
            batch["images"].to(device), image_ids=batch["image_ids"].to(device),
            cache_dir=cache_dir, save_cache=True,
        )
        keep = r.abs().sum(-1) != 0
        s = r[keep].sum(0)
        total = s if total is None else total + s
        count += int(keep.sum())
    return total / max(count, 1)


def answer_prior(cfg: Dict[str, Any], samples: List[Dict[str, Any]]) -> List[List[str]]:
    """Finglish: question-only baseline — javab por-tekrar train baraye (rule, 2 kalame aval)."""
    train = split_rows(cfg, "train", cfg["train_captions_json"])
    by_key, by_rule = defaultdict(Counter), defaultdict(Counter)
    for r in train:
        a = " ".join(tok(str(r["answer"])))
        if a in NON_LEXICAL_ANSWERS:
            continue
        q = tok(r["question"])
        by_key[(r.get("rule"), tuple(q[:2]))][a] += 1
        by_rule[r.get("rule")][a] += 1
    out = []
    for s in samples:
        c = by_key.get((s.get("rule"), tuple(tok(s["question"])[:2]))) or by_rule.get(s.get("rule"))
        out.append(c.most_common(1)[0][0].split() if c else [])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Image-ablation diagnostic for QD captioners")
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--modes", default="normal,shuffle,mean,mask")
    p.add_argument("--beam", type=int, default=3)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    bench_cfg = load_config(str(config_path))
    resolve_path_fields(bench_cfg, PATH_KEYS)
    seed = int(bench_cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(Path(args.ckpt).resolve(), map_location="cpu", weights_only=False)
    model_cfg = dict(state.get("config") or {})
    cfg = {**model_cfg, **{k: bench_cfg[k] for k in DATA_KEYS if k in bench_cfg}}
    cfg.setdefault("train_captions_json", bench_cfg["train_captions_json"])
    vocab, q_vocab = vocab_from_itos(state["vocab"]), vocab_from_itos(state["q_vocab"])
    model = build_captioner(cfg, vocab, q_vocab)
    load_model_state(model, state["model"])
    model.to(device).eval()
    model_class = cfg.get("model_class", "QuestionDependentImageCaptioner")

    ds = build_split_dataset(cfg, args.split, vocab, q_vocab)
    loader = DataLoader(ds, batch_size=int(cfg.get("batch_size", 16)), shuffle=False,
                        num_workers=0, collate_fn=collate_batch)
    cache_dir = region_cache_dir_for_split(cfg, args.split)
    samples = ds.samples
    id_map = derangement(sorted({int(s["image_id"]) for s in samples}), seed)
    mean_vec = split_mean_region(model, loader, device, cache_dir)

    results: Dict[str, Any] = {}
    hyps_by_mode: Dict[str, List[List[str]]] = {}
    for mode in [m for m in args.modes.split(",") if m]:
        if mode == "mask" and model_class != "MMTransformerCopyCaptioner":
            continue
        set_seed(seed)
        patch_regions(model, mode, id_map, mean_vec)
        hyps, _ = generate_split_captions(model, loader, vocab.itos, device,
                                          int(cfg["max_caption_len"]), beam_size=args.beam,
                                          region_cache_dir=cache_dir)
        hyps_by_mode[mode] = hyps
        per_rule = defaultdict(list)
        for i, s in enumerate(samples):
            per_rule[s["rule"] or "unknown"].append(i)
        results[mode] = {
            "overall": score_block(hyps, samples),
            "per_rule": {r: score_block([hyps[i] for i in idx], [samples[i] for i in idx])
                         for r, idx in sorted(per_rule.items())},
        }
        if mode != "normal" and "normal" in hyps_by_mode:
            base = hyps_by_mode["normal"]
            results[mode]["changed_vs_normal"] = sum(a != b for a, b in zip(base, hyps)) / len(hyps)
        print(f"{mode:8s} " + "  ".join(f"{k}={v:.4f}" for k, v in results[mode]["overall"].items()
                                       if k in ("BLEU-4", "CIDEr", "AnswerHit")))

    prior = answer_prior(cfg, samples)
    results["prior"] = {"overall": score_block(prior, samples)}
    print(f"prior    AnswerHit={results['prior']['overall']['AnswerHit']:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "ckpt": str(Path(args.ckpt).resolve()), "model_class": model_class,
        "seed": model_cfg.get("seed"), "split": args.split, "beam": args.beam,
        "results": results,
        "examples": [
            {"question": s["question"], "answer": s["answer"], "gt": s["caption"],
             **{m: " ".join(h[i]) for m, h in hyps_by_mode.items()}}
            for i, s in enumerate(samples[:60])
        ],
    }, indent=1), encoding="utf-8")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
