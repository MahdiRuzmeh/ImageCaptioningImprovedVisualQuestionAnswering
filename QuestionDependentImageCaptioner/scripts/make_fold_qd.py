"""Cross-fitting: QD caption data of ONE fold of train2014 images → JSON + BLIP training config for that fold.

Finglish: fold assignment az ``CaptionVQA/data/crossfit_folds.json`` (hamoon file-e VQA). Captioner-e fold k
faghat row haye QD (verdict == ok) ke image-eshoon dar fold k-e train mishe; akharin ~val_rows row (snap be
marz-e image) = val baraye entekhab-e epoch. Caption-e image haye fold k ro ba'dan model-e fold 1 − k minevise.

    python scripts/make_fold_qd.py --qd <QD json> --folds ../CaptionVQA/data/crossfit_folds.json --fold 0 \
        --out-json /kaggle/working/qd_fold0.json --base-config configs/b_blip_full.yaml \
        --out-config /kaggle/working/configs/b_blip_fold0.yaml --save-dir /kaggle/working/model_b/b_blip_fold0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import load_qd_json, snap_row_range  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--qd", required=True)
    p.add_argument("--folds", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--val-rows", type=int, default=1500)
    p.add_argument("--out-json", required=True)
    p.add_argument("--base-config", default="configs/b_blip_full.yaml")
    p.add_argument("--out-config", required=True)
    p.add_argument("--save-dir", required=True)
    args = p.parse_args()

    folds = json.loads(Path(args.folds).read_text(encoding="utf-8"))["train2014"]
    rows = [r for r in load_qd_json(args.qd) if folds.get(str(r["image_id"])) == args.fold]
    n = len(rows)
    tr = snap_row_range(rows, [0, n - args.val_rows])
    va = snap_row_range(rows, [n - args.val_rows, n])
    imgs = {r["image_id"] for r in rows}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps({"info": {"fold": args.fold, "rows": n, "images": len(imgs),
                                                        "train_row_range": list(tr), "val_row_range": list(va)},
                                               "annotations": rows}, ensure_ascii=False), encoding="utf-8")
    cfg = yaml.safe_load(open(PROJECT_ROOT / args.base_config, encoding="utf-8"))
    cfg.update(train_captions_json=args.out_json, train_row_range=list(tr), val_row_range=list(va),
               save_dir=args.save_dir)
    cfg.pop("test_row_range", None)
    Path(args.out_config).parent.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(args.out_config, "w", encoding="utf-8"), sort_keys=False)
    print(f"fold {args.fold}: {n} QD rows on {len(imgs)} images; train {tr}, val {va} -> {args.out_json}")
    print(f"config -> {args.out_config}")


if __name__ == "__main__":
    main()
