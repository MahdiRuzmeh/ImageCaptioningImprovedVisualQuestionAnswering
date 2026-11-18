"""BUTD Faster R-CNN region features for a VQA split → work_dir/features/regions_{split}/ (packed memmap).

Finglish: hamoon detector-e frozen-e captioner A (Visual Genome, 36 detection + 1 mean token, 2048-D).
Format = packed cache-e captioner (QuestionDependentImageCaptioner/packed_cache.py): feats.f16 (N, 37, 2048)
memmap + done flag → chand process (2 GPU) mitoonan hamzaman benevisan, train_vqa.py mostaghim az memmap
mikhoone (RAM-e process ha share mishe).

  --packed-dir <butd_packed>  : feature haye mojood (cache-e model A, 79,568 image) copy mishan, extract nemishan
  --shard i/n --device cuda:i : extract-e image haye baghi mande (har GPU yek shard)

    python prepare_regions.py --config configs/blip.yaml                               # local, 1 GPU
    python prepare_regions.py --config C --init-only --packed-dir /kaggle/input/.../butd_packed   # 1) init + copy
    python prepare_regions.py --config C --shard 0/2 --device cuda:0 &                                # 2) GPU 0
    python prepare_regions.py --config C --shard 1/2 --device cuda:1 &                                #    GPU 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import CAPTIONER_ROOT, load_config, read_json

# Finglish: captioner bayad aval-e sys.path bashe (train.py / models/ ba CaptionVQA esm-e moshtarak daran)
sys.path.insert(0, str(CAPTIONER_ROOT))
from packed_cache import PackedWriter, init_packed, is_packed  # noqa: E402

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


def copy_from_packed(src_dir: Path, writer: PackedWriter, ids) -> int:
    """Finglish: feature haye cache-e model A (hamoon detector) → bedoon extract-e dobare."""
    import json
    meta = json.loads((src_dir / "meta.json").read_text())
    n, t, d = len(meta["ids"]), int(meta["tokens"]), int(meta["feat_dim"])
    feats = np.memmap(src_dir / "feats.f16", np.float16, "r", shape=(n, t, d))
    boxes = np.memmap(src_dir / "boxes.f16", np.float16, "r", shape=(n, t, 4))
    ntok = np.memmap(src_dir / "ntok.i16", np.int16, "r", shape=(n,))
    done = np.memmap(src_dir / "done.u8", np.uint8, "r", shape=(n,))
    src = {int(i): k for k, i in enumerate(meta["ids"])}
    todo = [i for i in writer.todo(ids) if i in src and done[src[i]]]
    for iid in tqdm(todo, desc="copy from packed cache", mininterval=30):
        k = src[iid]
        m = int(ntok[k])
        writer.write(iid, np.asarray(feats[k, :m]), np.asarray(boxes[k, :m]))
    writer.flush()
    return len(todo)


def features_root(cfg) -> Path:
    """Finglish: features_dir (config) — Kaggle: /tmp (feature haye Stage B ~18 GB, /kaggle/working 20 GB dare)."""
    return Path(cfg["features_dir"]) if cfg.get("features_dir") else Path(cfg["work_dir"]) / "features"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/blip.yaml")
    p.add_argument("--splits", default="train,dev,test")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default=None, help="e.g. cuda:1")
    p.add_argument("--shard", default="0/1", help="i/n: this process extracts the i-th of n parts")
    p.add_argument("--packed-dir", nargs="*", default=None,
                   help="existing BUTD packed caches to copy from (e.g. qd-butd-packed and a val2014 cache)")
    p.add_argument("--init-only", action="store_true", help="create the files (+ copy from --packed-dir) and exit")
    args = p.parse_args()
    cfg = load_config(args.config)
    work = Path(cfg["work_dir"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    T, D = int(cfg["max_regions"]), int(cfg["region_dim"])
    si, sn = (int(x) for x in args.shard.split("/"))
    run = None
    for split in args.splits.split(","):
        out = features_root(cfg) / f"regions_{split}"
        recs = read_json(work / "split" / f"{split}.json")
        ids_sp = sorted({(int(r["image_id"]), r["image_split"]) for r in recs})
        sp_of = dict(ids_sp)
        ids = [i for i, _ in ids_sp]
        init_packed(out, ids, T, D)  # Finglish: age hast va yeksan-e → resume
        writer = PackedWriter(out)
        for src in args.packed_dir or []:
            if is_packed(src):
                print(f"{split}: copied {copy_from_packed(Path(src), writer, ids)} images from {src}", flush=True)
        if args.init_only:
            print(f"{split}: {len(ids)} images, still to extract {len(writer.todo(ids))}")
            continue
        todo = writer.todo(ids[si::sn])
        print(f"{split} shard {si}/{sn}: {len(ids[si::sn])} images, to extract {len(todo)} on {device}")
        if not todo:
            continue
        if run is None:
            from scripts.extract_visual import build_butd
            run = build_butd({"max_regions": T, "visual_model": str(CAPTIONER_ROOT / "outputs" / "butd_weights")}, device)
        items = [(iid, Path(cfg[f"{sp_of[iid]}_images_dir"]) / TEMPLATE[sp_of[iid]].format(iid)) for iid in todo]
        dl = DataLoader(Images(items), batch_size=1, num_workers=args.num_workers, collate_fn=collate)
        for imgs, chunk in tqdm(dl, desc=f"{split} shard {si}", mininterval=30):
            f, b = run(imgs)
            writer.write(chunk[0], f[0][:T].half().cpu().numpy(), b[0][:T].half().cpu().numpy())
        writer.flush()
        print(f"{split} shard {si}/{sn}: done -> {out}")


if __name__ == "__main__":
    main()
