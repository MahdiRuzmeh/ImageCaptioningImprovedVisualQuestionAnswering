"""Full QD caption JSON for the captioner comparison (model A / model B use the same file + split).

Finglish:
    Voroodi = khorooji-e QuestionDependentCaptionGenerator (283,707 row, sorted by image_id).
    - Faghat row haye ``caption_quality_verdict == "ok"`` mimoonan (GRAMMAR_ISSUE_ONLY va WRONG hazf).
    - Tartib (sorted by image_id) hefz mishe → ``snap_row_range`` split-e image-disjoint mide.
    - Split: test = akharin ``--test-rows`` row, val = ``--val-rows`` row ghablesh, baghi train.
      Range ha (snap shode) dar ``info.splits`` neveshte mishan; config ha hamina ro estefade mikonan.

    python scripts/prepare_full_qd.py \
        --src ../QuestionDependentCaptionGenerator/outputs/vqa_v2_question_dependent_captions_train2014.json \
        --out ../dataset/vqa_v2_qd_captions_full_train2014.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import snap_row_range  # noqa: E402

KEEP_FIELDS = (
    "question_id", "image_id", "question", "answer", "answer_count", "answer_consensus",
    "caption", "caption_quality_verdict", "suspicious", "rule",
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--val-rows", type=int, default=3000)
    p.add_argument("--test-rows", type=int, default=10000)
    args = p.parse_args()

    data = json.loads(Path(args.src).read_text(encoding="utf-8"))
    rows = data["annotations"]
    ids = [int(r["image_id"]) for r in rows]
    assert ids == sorted(ids), "source rows must be sorted by image_id"
    verdicts = Counter(r.get("caption_quality_verdict") for r in rows)
    kept = [{k: r.get(k) for k in KEEP_FIELDS} for r in rows if r.get("caption_quality_verdict") == "ok"]

    n = len(kept)
    val_start = n - args.test_rows - args.val_rows
    tr = snap_row_range(kept, [0, val_start])
    va = snap_row_range(kept, [val_start, n - args.test_rows])
    te = snap_row_range(kept, [n - args.test_rows, n])
    splits = {"train_row_range": list(tr), "val_row_range": list(va), "test_row_range": list(te)}
    for name, (s, e) in splits.items():
        imgs = {kept[i]["image_id"] for i in range(s, e)}
        print(f"{name}: [{s}, {e}) rows={e - s} images={len(imgs)}")

    info = {
        "description": "VQA v2 QD captions, filtered for captioner training (caption_quality_verdict == ok)",
        "source": Path(args.src).name,
        "source_rows": len(rows),
        "source_verdicts": dict(verdicts),
        "num_samples": n,
        "num_images": len(set(r["image_id"] for r in kept)),
        "splits": splits,
        "source_info_versions": data.get("info", {}).get("versions"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"info": info, "annotations": kept}, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out} ({n} rows, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
