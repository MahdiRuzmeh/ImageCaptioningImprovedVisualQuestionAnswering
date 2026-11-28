"""Blinded human review of the question-dependent caption dataset: draw the sample and build the review page.

Stratified random sample of captions with the final verdict "ok": 300 without and 200 with the heuristic "suspicious"
flag (fixed seed, drawn before anyone looks at the captions). The items are shuffled together, so the reviewer cannot
tell the groups apart; group, verdict and flag are stored only in sample_key.json.

    python audit/human_review/make_review_sample.py \
        --qd ../dataset/vqa_v2_question_dependent_captions_train2014.json \
        --annotations ../dataset/v2_mscoco_train2014_annotations.json
Outputs (next to this script): sample_key.json (do not open while reviewing), review.html (open in a browser).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--qd", required=True, help="question-dependent caption dataset JSON (with caption_quality_verdict)")
    p.add_argument("--annotations", required=True, help="VQA v2 train2014 annotations (for the ten human answers)")
    p.add_argument("--n-unflagged", type=int, default=300)
    p.add_argument("--n-flagged", type=int, default=200)
    p.add_argument("--overlap", type=int, default=100, help="first N items are reviewed by both reviewers")
    p.add_argument("--seed", type=int, default=20261128)
    p.add_argument("--images-rel", default="../../../dataset/train2014",
                   help="train2014 image folder, relative to review.html")
    args = p.parse_args()

    rows = json.loads(Path(args.qd).read_text(encoding="utf-8"))["annotations"]
    ok = [r for r in rows if r.get("caption_quality_verdict") == "ok"]
    groups = {"unflagged": [r for r in ok if not r.get("suspicious")], "flagged": [r for r in ok if r.get("suspicious")]}
    rng = random.Random(args.seed)
    picked = [("unflagged", r) for r in rng.sample(groups["unflagged"], args.n_unflagged)] + \
             [("flagged", r) for r in rng.sample(groups["flagged"], args.n_flagged)]
    rng.shuffle(picked)

    ann = {a["question_id"]: a for a in json.loads(Path(args.annotations).read_text(encoding="utf-8"))["annotations"]}
    items, key = [], []
    for n, (g, r) in enumerate(picked, 1):
        a = ann[r["question_id"]]
        items.append({"item": n, "image": f"{args.images_rel}/COCO_train2014_{int(r['image_id']):012d}.jpg",
                      "question": r["question"], "answer": r["answer"],
                      "answers": [x["answer"] for x in a["answers"]], "caption": r["caption"]})
        key.append({"item": n, "question_id": r["question_id"], "image_id": r["image_id"], "group": g,
                    "suspicious_reasons": r.get("suspicious_reasons", []), "rule": r.get("rule")})
    meta = {"seed": args.seed, "population": {g: len(v) for g, v in groups.items()},
            "sample": {"unflagged": args.n_unflagged, "flagged": args.n_flagged}, "overlap_items": args.overlap}
    (HERE / "sample_key.json").write_text(json.dumps({"meta": meta, "items": key}, indent=1), encoding="utf-8")
    page = (HERE / "review_template.html").read_text(encoding="utf-8")
    page = page.replace("/*ITEMS*/[]", json.dumps(items, ensure_ascii=False)).replace("/*OVERLAP*/0", str(args.overlap))
    (HERE / "review.html").write_text(page, encoding="utf-8")
    print(f"population {meta['population']}; sample {len(items)} items (seed {args.seed}); overlap 1-{args.overlap}")
    print(f"-> {HERE / 'review.html'}  (key: sample_key.json, keep closed while reviewing)")


if __name__ == "__main__":
    main()
