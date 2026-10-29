"""Faster R-CNN region feature ha ro baraye hame split ha yek bar cache kon.

Finglish:
    ``images_from_cache: true`` dar train/benchmark JPEG load nemikone, pas
    ghabl az train bayad har image_id toye cache bashe. In script image haye
    train/val/test (row range ha) ro peyda mikone, oonaei ke cache nadaran ro
    ba RegionEncoder (detector freeze) encode mikone va raw ROI 1024-D save mikone.

    python scripts/cache_regions.py --config configs/bench_v1.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.captioner_v1 import RegionEncoder  # noqa: E402
from train import (  # noqa: E402
    load_config,
    region_cache_dir_for_split,
    resolve_path_fields,
    split_rows,
)


class _Images(Dataset):
    """Finglish: JPEG decode dar worker ha (ghablan serial dar main process)."""

    def __init__(self, ids, images_dir: Path, template: str, transform) -> None:
        self.ids, self.dir, self.template, self.tf = ids, images_dir, template, transform

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        iid = self.ids[i]
        img = Image.open(self.dir / self.template.format(image_id=iid)).convert("RGB")
        return self.tf(img), iid


def main() -> None:
    p = argparse.ArgumentParser(description="Cache Faster R-CNN regions")
    p.add_argument("--config", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0, help="JPEG decode workers")
    p.add_argument("--shard", default="0/1",
                   help="k/n: faghat image haye i %% n == k (2 GPU → 2 process)")
    p.add_argument("--device", default=None, help="e.g. cuda:1")
    args = p.parse_args()
    shard_k, shard_n = (int(x) for x in args.shard.split("/"))

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    cfg = load_config(str(config_path))
    resolve_path_fields(
        cfg,
        (
            "train_captions_json",
            "train_images_dir",
            "val_images_dir",
            "train_region_cache_dir",
            "val_region_cache_dir",
            "test_region_cache_dir",
        ),
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    size = int(cfg.get("image_size", 448))
    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    encoder = RegionEncoder(int(cfg["max_regions"]), int(cfg["region_dim"]))
    encoder.to(device).eval()

    for split in ("train", "val", "test"):
        if not cfg.get(f"{split}_row_range"):
            continue
        cache_dir = region_cache_dir_for_split(cfg, split)
        img_key = "train" if split == "train" else "val"
        images_dir = Path(cfg[f"{img_key}_images_dir"])
        template = cfg[f"{img_key}_image_filename_template"]
        ids = sorted(
            {int(r["image_id"]) for r in split_rows(cfg, split, cfg["train_captions_json"])}
        )
        todo = [
            i for n, i in enumerate(ids)
            if n % shard_n == shard_k and not (Path(cache_dir) / f"{i}.pt").exists()
        ]
        print(f"split={split} images={len(ids)} to_cache={len(todo)} "
              f"shard={args.shard} dir={cache_dir}")
        ds = _Images(todo, images_dir, template, transform)
        dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers)
        for imgs, chunk in tqdm(dl, desc=f"{split}[{args.shard}]"):
            with torch.no_grad():
                encoder(
                    imgs.to(device),
                    image_ids=chunk.to(device),
                    cache_dir=cache_dir,
                    save_cache=True,
                )


if __name__ == "__main__":
    main()
