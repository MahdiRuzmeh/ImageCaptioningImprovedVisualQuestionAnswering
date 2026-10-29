"""COCO human caption ha → JSON ba format-e QD (pretrain-e captioner, step 2).

Finglish:
    Har (image, COCO caption) yek row mishe: ``question`` sabet ("what is in the
    image"), ``caption`` = caption-e adam, ``answer`` = "". Rows sorted by image_id
    ta ``train_row_range`` / ``val_row_range`` (snap be marz image) kar kone.

    Leakage: image haye QD az ``--qd-train-end`` (row, snap shode) be ba'd —
    yani val/test-e captioner va rows-e estefade nashode — kharej mishan.
    CaptionVQA dev/test az val2014 hastan, pas inja (train2014) nistan.

    python scripts/build_coco_pretrain.py \
        --coco ../dataset/captions_train2014.json \
        --qd ../dataset/vqa_v2_qd_captions_enriched_train2014.json \
        --out ../dataset/coco_pretrain_captions_train2014.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import load_qd_json, snap_row_range  # noqa: E402

PRETRAIN_QUESTION = "what is in the image"


def main() -> None:
    p = argparse.ArgumentParser(description="Build COCO caption pretrain JSON (QD format)")
    p.add_argument("--coco", required=True, help="captions_train2014.json")
    p.add_argument("--qd", required=True, help="QD JSON (rows sorted by image_id)")
    p.add_argument("--qd-train-end", type=int, default=9500,
                   help="QD rows >= this (snapped) are held out → their images excluded")
    p.add_argument("--val-images", type=int, default=1000, help="COCO images for pretrain val")
    p.add_argument("--max-images", type=int, default=0, help=">0 → subsample (pilot)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    qd = load_qd_json(args.qd)
    cut = snap_row_range(qd, [args.qd_train_end, len(qd)])[0]
    held_out = {int(r["image_id"]) for r in qd[cut:]}

    with open(args.coco, encoding="utf-8") as f:
        coco = json.load(f)["annotations"]
    by_image: dict = {}
    for a in coco:
        iid = int(a["image_id"])
        if iid not in held_out:
            by_image.setdefault(iid, []).append(" ".join(str(a["caption"]).split()))

    # Finglish: val = image haye random (na akharin id ha) → tozie yeksan
    ids = sorted(by_image)
    rng = random.Random(args.seed)
    if args.max_images > 0:
        ids = sorted(rng.sample(ids, min(args.max_images, len(ids))))
    val_ids = set(rng.sample(ids, args.val_images))
    train_ids = [i for i in ids if i not in val_ids]

    def rows(id_list):
        return [
            {"image_id": i, "question": PRETRAIN_QUESTION, "caption": c, "answer": ""}
            for i in id_list for c in by_image[i]
        ]

    # Finglish: aval train (sorted) ba'd val (sorted) → row range ha disjoint image
    # (har image faghat yek taraf; snap dar marz tagheer nemide chon id ha joda an)
    train_rows, val_rows = rows(train_ids), rows(sorted(val_ids))
    out = {
        "info": {
            "source": "COCO captions_train2014 (human)",
            "question": PRETRAIN_QUESTION,
            "excluded_qd_images": len(held_out),
            "train_images": len(train_ids),
            "val_images": len(val_ids),
            "train_row_range": [0, len(train_rows)],
            "val_row_range": [len(train_rows), len(train_rows) + len(val_rows)],
        },
        "annotations": train_rows + val_rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(json.dumps(out["info"], indent=1))


if __name__ == "__main__":
    main()
