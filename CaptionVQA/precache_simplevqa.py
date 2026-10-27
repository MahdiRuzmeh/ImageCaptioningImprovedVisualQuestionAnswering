"""Fill SimpleVQA's own region + ResNet-101 caches with small batches (8 GB GPU friendly).

SimpleVQA extracts these lazily inside its first training epoch at batch 32, which
overflows an 8 GB GPU. Same code path (VQAModel._regions / _global_feat), so the
cached tensors are exactly what SimpleVQA would compute itself.

Run from SimpleVQA/:
    python ../CaptionVQA/precache_simplevqa.py --config configs/caption_vqa_small.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import read_json  # noqa: E402
sys.path.remove(str(HERE))  # CaptionVQA/models.py != captioner "models" package

sys.path.insert(0, str(HERE.parent / "SimpleVQA"))
import train as svqa  # noqa: E402


class _ImageDS(Dataset):
    """Hamoon transform-e SimpleVQA VQADataset (resize 448 + ImageNet normalize)."""

    def __init__(self, ids, images_dir: str, template: str, size: int) -> None:
        self.ids, self.dir, self.template = ids, Path(images_dir), template
        self.tf = svqa.transforms.Compose([
            svqa.transforms.Resize((size, size)),
            svqa.transforms.ToTensor(),
            svqa.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        iid = self.ids[i]
        return self.tf(Image.open(self.dir / self.template.format(image_id=iid)).convert("RGB")), iid


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/caption_vqa_small.yaml")
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()
    cfg = svqa.load_config(args.config)
    svqa.resolve_path_fields(cfg, svqa.PATH_KEYS)
    device = torch.device("cuda")
    model = svqa.VQAModel(10, 10, 0, None, use_captioner=False).to(device).eval()
    split_dir = HERE / "outputs" / "small" / "split"
    for img_split, names, tmpl in (("train", ["train"], "COCO_train2014_{image_id:012d}.jpg"),
                                   ("val", ["dev", "test"], "COCO_val2014_{image_id:012d}.jpg")):
        reg_dir = svqa.region_cache_dir_for_split(cfg, img_split)
        glob_dir = svqa.global_cache_dir_for_split(cfg, img_split)
        ids = sorted({r["image_id"] for n in names for r in read_json(split_dir / f"{n}.json")})
        todo = [i for i in ids if not (Path(reg_dir) / f"{i}_k32_raw1024.pt").exists()
                or not (Path(glob_dir) / f"{i}_resnet101.pt").exists()]
        print(f"{img_split}: {len(ids)} images, {len(todo)} to cache")
        if not todo:
            continue
        ds = _ImageDS(todo, cfg[f"{img_split}_images_dir"], tmpl, int(cfg["image_size"]))
        with torch.no_grad():
            for imgs, iids in tqdm(DataLoader(ds, batch_size=args.batch_size, num_workers=4), desc=img_split):
                imgs, iids = imgs.to(device), iids.to(device)
                model._regions(imgs, image_ids=iids, cache_dir=reg_dir, save_cache=True)
                model._global_feat(imgs, image_ids=iids, cache_dir=glob_dir, save_cache=True)


if __name__ == "__main__":
    main()
