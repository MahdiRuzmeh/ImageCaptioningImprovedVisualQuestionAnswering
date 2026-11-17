"""Write BLIP question-dependent captions for the CaptionVQA questions (train2014 + val2014 images).

Finglish: model-e B (train_blip.py, best.pt) baraye har (soal, tasvir) dar
``CaptionVQA/data/blip_caption_requests.json`` caption minevise (beam 3, top-1).
Tasvir: image_split "train" → train2014, "val" → val2014. ``--shard i/n`` baraye 2 GPU.
Khorooji: {question_id: caption} JSON (har shard joda; --merge hame ro yeki mikone).

    python scripts/blip_caption_vqa.py --config configs/b_blip_full.yaml --ckpt best.pt \
        --requests ../CaptionVQA/data/blip_caption_requests.json --val-images-dir <val2014> \
        --shard 0/2 --device cuda:0 --out captions_shard0.json
    python scripts/blip_caption_vqa.py --merge captions_shard0.json captions_shard1.json --out captions_blip.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import load_config, resolve_path_fields  # noqa: E402
from train_blip import QdBlipDataset, build_model, decode_split  # noqa: E402


class VqaImages(QdBlipDataset):
    """Finglish: hamoon preprocess-e BLIP, vali tasvir az train2014 ya val2014 (image_split)."""

    def __init__(self, rows, cfg, tokenizer, bos, sep, dirs):
        super().__init__(rows, cfg, tokenizer, bos, sep)
        self.dirs = dirs  # split → (dir, template)

    def __getitem__(self, i):
        r = self.rows[i]
        d, tmpl = self.dirs[r["image_split"]]
        img = Image.open(d / tmpl.format(image_id=int(r["image_id"]))).convert("RGB")
        return {"pixel_values": self.tf(img), "prompt": self.prompt_ids[i], "caption": [self.sep], "idx": i}


def merge(paths, out) -> None:
    cap = {}
    for p in paths:
        cap.update(json.loads(Path(p).read_text(encoding="utf-8")))
    Path(out).write_text(json.dumps(cap, ensure_ascii=False), encoding="utf-8")
    print(f"merged {len(cap)} captions -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config")
    p.add_argument("--ckpt")
    p.add_argument("--requests")
    p.add_argument("--val-images-dir")
    p.add_argument("--shard", default="0/1")
    p.add_argument("--device", default="cuda")
    p.add_argument("--beam", type=int, default=3)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--chunk", type=int, default=10000, help="save after every N captions (resume)")
    p.add_argument("--caption-model", type=int, default=None,
                   help="cross-fitting: only requests with caption_model == k (this fold model never saw their images)")
    p.add_argument("--merge", nargs="*")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.merge:
        merge(args.merge, args.out)
        return

    cfg = load_config(str(Path(args.config) if Path(args.config).is_absolute() else PROJECT_ROOT / args.config))
    resolve_path_fields(cfg, ("train_images_dir",))
    if cfg.get("blip_model") and not str(cfg["blip_model"]).startswith("Salesforce/"):
        resolve_path_fields(cfg, ("blip_model",))
    reqs = json.loads(Path(args.requests).read_text(encoding="utf-8"))
    if args.caption_model is not None:
        reqs = [r for r in reqs if int(r["caption_model"]) == args.caption_model]
    i, n = (int(x) for x in args.shard.split("/"))
    rows = [{**r, "caption": ""} for r in reqs[i::n]]
    if args.limit:
        rows = rows[: args.limit]
    device = torch.device(args.device)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model, tokenizer = build_model(cfg, device, state["model"])
    model.eval()
    dirs = {"train": (Path(cfg["train_images_dir"]), "COCO_train2014_{image_id:012d}.jpg"),
            "val": (Path(args.val_images_dir), "COCO_val2014_{image_id:012d}.jpg")}
    for sp, (d, tmpl) in dirs.items():
        if any(r["image_split"] == sp for r in rows):
            first = next(r for r in rows if r["image_split"] == sp)
            assert (d / tmpl.format(image_id=int(first["image_id"]))).exists(), f"{sp} images not found in {d}"
    # Finglish: chunk be chunk + save ba'd az har chunk → resume age session ghat beshe (khorooji ro attach kon)
    out_path = Path(args.out)
    out = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    todo = [r for r in rows if str(r["question_id"]) not in out]
    print(f"shard {i}/{n}: {len(rows)} requests, {len(out)} already captioned, {len(todo)} to do", flush=True)
    t0 = time.time()
    for k in range(0, len(todo), args.chunk):
        part = todo[k : k + args.chunk]
        ds = VqaImages(part, cfg, tokenizer, model.bos, model.sep, dirs)
        preds = decode_split(model, ds, tokenizer, cfg, device, args.beam, 0, 1, False)
        out.update({str(r["question_id"]): c for r, c in zip(part, preds)})
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        tmp.replace(out_path)
        done = k + len(part)
        rate = done / max(time.time() - t0, 1e-9)
        print(f"  {done}/{len(todo)} captions ({rate:.1f}/s, ~{(len(todo) - done) / max(rate, 1e-9) / 60:.0f} min left)",
              flush=True)
    print(f"shard {i}/{n}: {len(out)} captions in {time.time() - t0:.0f}s -> {args.out}")
    for r in rows[:5]:
        print(f"  Q: {r['question']}  ->  {out[str(r['question_id'])]}")


if __name__ == "__main__":
    main()
