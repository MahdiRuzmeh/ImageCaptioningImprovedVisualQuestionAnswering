"""Build the image-disjoint VQA v2 subset: train (train2014) / dev + test (val2014).

Finglish: image haye QD caption dataset (ke captioner roshon train shode) az VQA
train hazf mishan, ta caption-e train va test keyfiat-e yeksan dashte bashan.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from common import load_config, read_json, write_json


def take_images(by_image: Dict[int, List[dict]], image_ids: List[int], n_questions: int) -> List[int]:
    out: List[int] = []
    for iid in image_ids:
        if len(out) >= n_questions:
            break
        out.extend(int(q["question_id"]) for q in by_image[iid])
    return out


def load_split(questions_json: str, annotations_json: str) -> tuple:
    qs = read_json(questions_json)["questions"]
    anns = {int(a["question_id"]): a for a in read_json(annotations_json)["annotations"]}
    by_image: Dict[int, List[dict]] = defaultdict(list)
    for q in qs:
        if int(q["question_id"]) in anns:
            by_image[int(q["image_id"])].append(q)
    return qs, anns, by_image


def records(qids: List[int], qmap: Dict[int, dict], anns: Dict[int, dict], split: str) -> List[dict]:
    out = []
    for qid in qids:
        q, a = qmap[qid], anns[qid]
        out.append({
            "question_id": qid,
            "image_id": int(q["image_id"]),
            "image_split": split,
            "question": q["question"],
            "answers": [x["answer"] for x in a["answers"]],
            "answer_type": a["answer_type"],
            "question_type": a["question_type"],
        })
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/small.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    rng = random.Random(int(cfg["seed"]))

    qd_images = {int(r["image_id"]) for r in read_json(cfg["qd_captions_json"])["annotations"]}

    tr_qs, tr_anns, tr_by_img = load_split(cfg["train_questions_json"], cfg["train_annotations_json"])
    tr_imgs = sorted(i for i in tr_by_img if i not in qd_images)
    rng.shuffle(tr_imgs)
    tr_qids = take_images(tr_by_img, tr_imgs, int(cfg["num_train_questions"]))

    va_qs, va_anns, va_by_img = load_split(cfg["val_questions_json"], cfg["val_annotations_json"])
    va_imgs = sorted(va_by_img)
    rng.shuffle(va_imgs)
    dev_qids = take_images(va_by_img, va_imgs, int(cfg["num_dev_questions"]))
    qmap_va = {int(q["question_id"]): q for q in va_qs}
    dev_img_set = {int(qmap_va[qid]["image_id"]) for qid in dev_qids}
    rest = [i for i in va_imgs if i not in dev_img_set]
    test_qids = take_images(va_by_img, rest, int(cfg["num_test_questions"]))

    qmap_tr = {int(q["question_id"]): q for q in tr_qs}
    work = Path(cfg["work_dir"])
    split_recs = {
        "train": records(tr_qids, qmap_tr, tr_anns, "train"),
        "dev": records(dev_qids, qmap_va, va_anns, "val"),
        "test": records(test_qids, qmap_va, va_anns, "val"),
    }
    for name, recs in split_recs.items():
        write_json(work / "split" / f"{name}.json", recs)
        n_img = len({r["image_id"] for r in recs})
        print(f"{name}: {len(recs)} questions, {n_img} images")
    tr_img = {r["image_id"] for r in split_recs["train"]}
    print(f"train overlap with QD-captioner images = {len(tr_img & qd_images)} (must be 0)")


if __name__ == "__main__":
    main()
