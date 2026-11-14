"""Kaggle: QD caption JSON — path-e config, ya search dar /kaggle/input (row count = split-e config).

Finglish: momkene chand file ba esm-e moshabeh attach bashe (e.g. JSON-e ghadimi-e 406k bedoon
caption_quality_verdict). Faghat fili ke ba filter-e load_qd_json dagighan ``test_row_range[1]``
row dashte bashe ghabool mishe → split ba local yeki-e. Path ro print mikone; peyda nashe exit 1.

    QD=$(python scripts/kaggle_find_qd.py configs/a_full_butd.yaml)
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import load_qd_json  # noqa: E402

NAMES = ("vqa_v2_qd_captions_full_train2014.json", "vqa_v2_question_dependent_captions_train2014.json")


def main() -> None:
    cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
    root = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/kaggle/input")
    want = int(cfg["test_row_range"][1])
    seen = []
    # Finglish: aval path-e config (age hast), bad search
    own = Path(str(cfg.get("train_captions_json", "")))
    if own.is_file():
        n = len(load_qd_json(str(own)))
        if n == want:
            print(own)
            return
        seen.append(f"{own} ({n} rows after the ok filter; config path)")
    for name in NAMES:
        for p in sorted(root.rglob(name)):
            n = len(load_qd_json(str(p)))
            seen.append(f"{p} ({n} rows after the ok filter)")
            if n == want:
                print(p)
                return
    print(f"No QD JSON with {want} rows (verdict == ok) under {root}. Found:\n  " + "\n  ".join(seen or ["nothing"]),
          file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
