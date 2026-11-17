"""Cross-fitting fold assignment: every train2014 image → fold 0 or 1 (half / half, seeded).

Finglish: yek manba-e vahed baraye har do taraf:
  - captioner-e fold k (QuestionDependentImageCaptioner/scripts/make_fold_qd.py) faghat roye QD row haye
    image haye fold k train mishe;
  - caption-e har soal-e VQA az model-i miad ke oon image ro **nadide**: train2014 image dar fold k →
    caption_model = 1 − k. Image haye val2014 (dev/test) ham nesf-nesf beyn-e do model (caption_model = fold).
Khorooji: data/crossfit_folds.json {"train2014": {image_id: fold}, "val2014": {image_id: fold}} (commit mishe).

    python make_crossfit_folds.py --config configs/blip_full.yaml
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from common import load_config, read_json, write_json


def halves(ids, seed: int) -> dict:
    ids = sorted(ids)
    random.Random(seed).shuffle(ids)
    return {str(i): (0 if k < len(ids) // 2 else 1) for k, i in enumerate(ids)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/blip_full.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    work = Path(cfg["work_dir"])
    tr = {int(q["image_id"]) for q in read_json(cfg["train_questions_json"])["questions"]}
    va = {r["image_id"] for s in ("dev", "test") for r in read_json(work / "split" / f"{s}.json")}
    out = {"seed": int(cfg["seed"]), "train2014": halves(tr, int(cfg["seed"])), "val2014": halves(va, int(cfg["seed"]) + 1)}
    write_json(cfg["crossfit_folds_json"], out)
    for k in ("train2014", "val2014"):
        n1 = sum(out[k].values())
        print(f"{k}: {len(out[k])} images, fold 0 = {len(out[k]) - n1}, fold 1 = {n1}")
    print(f"-> {cfg['crossfit_folds_json']}")


if __name__ == "__main__":
    main()
