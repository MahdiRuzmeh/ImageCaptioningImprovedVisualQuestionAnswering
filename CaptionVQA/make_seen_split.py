"""Build the "seen" VQA v2 subset: all questions on the images the QD captioner was trained on.

Finglish: bar khalaf-e make_split.py (ke image haye captioner ro hazf mikone), inja
FAGHAT image haye train-e captioner (rows ``captioner_train_row_range`` az QD JSON,
snap be marz image_id) estefade mishan → caption ha aksaran javab ro daran.
Image ha be train / dev / test taghsim mishan (image-disjoint). Har record
``qd_pair`` dare: True age hamin (image, soal) dar train-e captioner bude.
Region cache-e captioner ham kopi mishe (hamoon feature haye train-e captioner).

    python make_seen_split.py --config configs/seen.yaml
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from common import load_config, read_json, write_json
from make_split import load_split, records


def snap(rows: list, i: int) -> int:
    """Hamoon snap_row_range-e captioner: marz jolo mire ta yek image nashkane."""
    i = max(0, min(i, len(rows)))
    while 0 < i < len(rows) and int(rows[i]["image_id"]) == int(rows[i - 1]["image_id"]):
        i += 1
    return i


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/seen.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    rng = random.Random(int(cfg["seed"]))

    qd_rows = read_json(cfg["qd_captions_json"])["annotations"]
    s, e = cfg["captioner_train_row_range"]
    cap_rows = qd_rows[snap(qd_rows, s): snap(qd_rows, e)]
    cap_imgs = sorted({int(r["image_id"]) for r in cap_rows})
    qd_pairs = {int(r["question_id"]) for r in cap_rows}
    print(f"captioner train: {len(cap_rows)} rows, {len(cap_imgs)} images")

    qs, anns, by_img = load_split(cfg["train_questions_json"], cfg["train_annotations_json"])
    imgs = [i for i in cap_imgs if i in by_img]
    rng.shuffle(imgs)
    n_dev, n_test = int(cfg["num_dev_images"]), int(cfg["num_test_images"])
    parts = {"dev": imgs[:n_dev], "test": imgs[n_dev:n_dev + n_test], "train": imgs[n_dev + n_test:]}
    qmap = {int(q["question_id"]): q for q in qs}
    work = Path(cfg["work_dir"])
    for name, part in parts.items():
        qids = [int(q["question_id"]) for i in sorted(part) for q in by_img[i]]
        recs = records(qids, qmap, anns, "train")
        for r in recs:
            r["qd_pair"] = r["question_id"] in qd_pairs
        write_json(work / "split" / f"{name}.json", recs)
        print(f"{name}: {len(recs)} questions, {len(part)} images, "
              f"{sum(r['qd_pair'] for r in recs)} captioner-trained (image, question) pairs")

    src, dst = Path(cfg["captioner_region_cache_dir"]), work / "region_cache" / "train"
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for i in imgs:
        if not (dst / f"{i}.pt").exists():
            shutil.copyfile(src / f"{i}.pt", dst / f"{i}.pt")
            n += 1
    print(f"copied {n} region-cache files from the captioner cache")


if __name__ == "__main__":
    main()
