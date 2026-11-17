"""BUTD Faster R-CNN region features for the VQA split → work_dir/features/regions_{split}.pt.

Finglish: hamoon detector-e frozen-e captioner A (Visual Genome, 36 detection + 1 mean token, 2048-D),
standard-e Bottom-Up Top-Down VQA. Khorooji: {"image_ids": [...], "feats": (N, 37, 2048) fp16, "boxes": (N, 37, 4)}.
Pad token = 0 (mask = abs-sum == 0). Har split joda save mishe → resume (split-e tamoom shode skip).

    python prepare_regions.py --config configs/blip.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import CAPTIONER_ROOT, load_config, read_json

# Finglish: captioner bayad aval-e sys.path bashe (train.py / models/ ba CaptionVQA esm-e moshtarak daran)
sys.path.insert(0, str(CAPTIONER_ROOT))
from scripts.extract_visual import build_butd  # noqa: E402

TEMPLATE = {"train": "COCO_train2014_{:012d}.jpg", "val": "COCO_val2014_{:012d}.jpg"}


class Images(Dataset):
    def __init__(self, items):
        self.items = items  # (image_id, path)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        iid, path = self.items[i]
        return Image.open(path).convert("RGB"), iid


def collate(batch):
    """Finglish: module-level (Windows spawn → lambda pickle nemishe)."""
    return [x[0] for x in batch], [x[1] for x in batch]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/blip.yaml")
    p.add_argument("--splits", default="train,dev,test")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default=None, help="e.g. cuda:1 (Kaggle: one split per GPU)")
    args = p.parse_args()
    cfg = load_config(args.config)
    work = Path(cfg["work_dir"])
    (work / "features").mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    T, D = int(cfg["max_regions"]), int(cfg["region_dim"])
    run = None
    for split in args.splits.split(","):
        out = work / "features" / f"regions_{split}.pt"
        if out.exists():
            print(f"{out} exists, skip")
            continue
        recs = read_json(work / "split" / f"{split}.json")
        ids = sorted({(r["image_id"], r["image_split"]) for r in recs})
        items = [(iid, Path(cfg[f"{sp}_images_dir"]) / TEMPLATE[sp].format(iid)) for iid, sp in ids]
        if run is None:
            run = build_butd({"max_regions": T, "visual_model": str(CAPTIONER_ROOT / "outputs" / "butd_weights")}, device)
        feats = torch.zeros(len(items), T, D, dtype=torch.float16)
        boxes = torch.zeros(len(items), T, 4, dtype=torch.float16)
        dl = DataLoader(Images(items), batch_size=1, num_workers=args.num_workers,
                        collate_fn=collate)
        k = 0
        for imgs, _ in tqdm(dl, desc=split, mininterval=30):
            f, b = run(imgs)
            n = min(len(f[0]), T)
            feats[k, :n] = f[0][:n].half().cpu()
            boxes[k, :n] = b[0][:n].half().cpu()
            k += 1
        torch.save({"image_ids": [iid for iid, _ in items], "feats": feats, "boxes": boxes}, out)
        print(f"{split}: {len(items)} images → {out} ({out.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
