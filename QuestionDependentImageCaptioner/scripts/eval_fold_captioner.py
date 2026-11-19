"""Cross-fitting: score a fold captioner on QD rows of the OTHER fold (images it never trained on).

Finglish: captioner-e fold k faghat roye image haye fold k train shode → roye row haye QD-e fold 1 − k test mishe
(hamoon kari ke dar CaptionVQA anjam mide: caption baraye image haye nadide). Ye sample-e sabet (seed) az n row.
Do halat:
  - ``--captions``: caption haye mojood (captions_crossfit.json, {question_id: caption}) → baseline, bedoon GPU;
  - ``--ckpt``: caption ba model-e jadid (beam 3); ba torchrun roye 2 GPU (decode_split shard + all_gather).
Metric: AnswerHit, YesNoPolarity, CIDEr/BLEU-4/METEOR/ROUGE-L (nesbat be caption-e QD) — kole + har answer type.

    python scripts/eval_fold_captioner.py --qd <QD json> --folds ../CaptionVQA/data/crossfit_folds.json --fold 0 \
        --captions ../CaptionVQA/vqa_strong_crossfit/captions/captions_crossfit.json --out fold0_current.json
    torchrun --nproc_per_node=2 scripts/eval_fold_captioner.py --qd <QD json> --folds ... --fold 0 \
        --config <fold config> --ckpt <save_dir>/best.pt --out fold0_new.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import load_config, load_qd_json, resolve_path_fields  # noqa: E402
from train_blip import score  # noqa: E402


def answer_type(a: str) -> str:
    a = a.strip().lower()
    if a in ("yes", "no"):
        return "yes/no"
    return "number" if a.isdigit() else "other"


def sample_rows(qd: str, folds_json: str, fold: int, n: int, seed: int):
    folds = json.loads(Path(folds_json).read_text(encoding="utf-8"))["train2014"]
    rows = [r for r in load_qd_json(qd) if folds.get(str(r["image_id"])) == 1 - fold]
    if n and n < len(rows):
        keep = sorted(random.Random(seed).sample(range(len(rows)), n))
        rows = [rows[i] for i in keep]
    return rows


def report(preds, rows, name: str) -> dict:
    overall = score(preds, rows)
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[answer_type(str(r["answer"]))].append(i)
    per_type = {k: score([preds[i] for i in v], [rows[i] for i in v]) for k, v in sorted(groups.items())}
    print(f"[{name}] {len(rows)} rows  " + "  ".join(f"{k}={v:.4f}" for k, v in overall.items() if k != "n"))
    for k, v in per_type.items():
        print(f"    {k:7s} n={int(v['n']):5d}  AnswerHit={v.get('AnswerHit', 0):.4f}  "
              f"YesNoPolarity={v.get('YesNoPolarity', 0):.4f}  CIDEr={v['CIDEr']:.4f}")
    return {"name": name, "overall": overall, "per_answer_type": per_type}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--qd", required=True)
    p.add_argument("--folds", required=True)
    p.add_argument("--fold", type=int, required=True, help="the captioner's own fold k (scored on fold 1 - k)")
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--captions", default=None, help="existing {question_id: caption} JSON (baseline, no GPU)")
    p.add_argument("--config", default=None)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--images-dir", default=None, help="train2014 images (default: the config's train_images_dir)")
    p.add_argument("--beam", type=int, default=3)
    p.add_argument("--name", default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rows = sample_rows(args.qd, args.folds, args.fold, args.n, args.seed)
    if args.captions:
        caps = json.loads(Path(args.captions).read_text(encoding="utf-8"))
        miss = [r for r in rows if str(r["question_id"]) not in caps]
        assert not miss, f"{len(miss)} sampled rows have no caption in {args.captions}"
        preds = [caps[str(r["question_id"])] for r in rows]
        rank, secs = 0, 0.0
    else:
        import torch
        import torch.distributed as dist
        from train_blip import QdBlipDataset, build_model, decode_split

        ddp_on = "RANK" in os.environ
        if ddp_on:
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")
        rank = dist.get_rank() if ddp_on else 0
        world = dist.get_world_size() if ddp_on else 1
        device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}" if torch.cuda.is_available() else "cpu")
        cfg = load_config(str(Path(args.config) if Path(args.config).is_absolute() else PROJECT_ROOT / args.config))
        if args.images_dir:
            cfg["train_images_dir"] = args.images_dir
        resolve_path_fields(cfg, ("train_images_dir",))
        if cfg.get("blip_model") and not str(cfg["blip_model"]).startswith("Salesforce/"):
            resolve_path_fields(cfg, ("blip_model",))
        cfg["answer_weight"] = 1.0  # Finglish: inja faghat generate → wazn lazem nist
        state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model, tokenizer = build_model(cfg, device, state["model"])
        model.eval()
        ds = QdBlipDataset(rows, cfg, tokenizer, model.bos, model.sep)
        t0 = time.time()
        preds = decode_split(model, ds, tokenizer, cfg, device, args.beam, rank, world, ddp_on)
        secs = time.time() - t0
        if ddp_on:
            dist.barrier()
            dist.destroy_process_group()
    if rank != 0:
        return
    res = report(preds, rows, args.name or Path(args.out).stem)
    res.update(fold=args.fold, scored_on_fold=1 - args.fold, n=len(rows), seed=args.seed, beam=args.beam,
               source=args.captions or args.ckpt, decode_secs=secs,
               predictions=[{"question_id": r["question_id"], "image_id": r["image_id"], "question": r["question"],
                             "answer": r["answer"], "gt": r["caption"], "pred": c} for r, c in zip(rows, preds)])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
