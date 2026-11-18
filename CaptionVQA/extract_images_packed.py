"""BUTD features for every image in a folder → one packed cache (same format as qd-butd-packed).

Finglish: baraye Stage B: hame-ye 40,504 tasvir-e val2014 yek bar extract mishan → khorooji-e notebook ro
dataset kon → notebook haye VQA copy mikonan (be jaye har bar ~2 saat extract). 2 GPU: --shard 0/2 va 1/2.

    python extract_images_packed.py --images-dir <val2014> --prefix COCO_val2014_ --out /kaggle/working/butd_val2014 --init-only
    python extract_images_packed.py --images-dir <val2014> --prefix COCO_val2014_ --out ... --shard 0/2 --device cuda:0 &
    python extract_images_packed.py --images-dir <val2014> --prefix COCO_val2014_ --out ... --shard 1/2 --device cuda:1 &
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import CAPTIONER_ROOT
from prepare_regions import Images, collate

sys.path.insert(0, str(CAPTIONER_ROOT))
from packed_cache import PackedWriter, init_packed  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--images-dir", required=True)
    p.add_argument("--prefix", default="COCO_val2014_")
    p.add_argument("--out", required=True)
    p.add_argument("--max-regions", type=int, default=37)
    p.add_argument("--init-only", action="store_true")
    p.add_argument("--shard", default="0/1")
    p.add_argument("--device", default=None)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    d = Path(args.images_dir)
    files = {int(f.stem.rsplit("_", 1)[-1]): f for f in d.glob(f"{args.prefix}*.jpg")}
    ids = sorted(files)[: args.limit] if args.limit else sorted(files)
    init_packed(args.out, ids, args.max_regions, 2048)
    if args.init_only:
        print(f"packed cache ready: {args.out} ({len(ids)} images in {d})")
        return
    si, sn = (int(x) for x in args.shard.split("/"))
    writer = PackedWriter(args.out)
    todo = writer.todo(ids[si::sn])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"shard {si}/{sn}: {len(ids[si::sn])} images, to extract {len(todo)} on {device}", flush=True)
    if not todo:
        return
    from scripts.extract_visual import build_butd
    run = build_butd({"max_regions": args.max_regions,
                      "visual_model": str(CAPTIONER_ROOT / "outputs" / "butd_weights")}, device)
    dl = DataLoader(Images([(i, files[i]) for i in todo]), batch_size=1, num_workers=args.num_workers,
                    collate_fn=collate)
    for imgs, chunk in tqdm(dl, desc=f"shard {si}", mininterval=60):
        f, b = run(imgs)
        writer.write(chunk[0], f[0][: args.max_regions].half().cpu().numpy(), b[0][: args.max_regions].half().cpu().numpy())
    writer.flush()
    print(f"shard {si}/{sn}: done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
