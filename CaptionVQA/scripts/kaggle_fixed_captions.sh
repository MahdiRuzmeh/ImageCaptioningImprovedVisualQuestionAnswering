#!/usr/bin/env bash
# Capacity control (ablation only): the same fixed caption ("an image.") for every train2014, val2014 and test2015
# question. CPU only, ~2 min. Run from the checkout:  cd <repo>/CaptionVQA && bash scripts/kaggle_fixed_captions.sh
# Inputs: mnhlfuch/vqa-v2 (train2014 + val2014 questions) and v2_OpenEnded_mscoco_test2015_questions.json.
# Output: /kaggle/working/fixed_captions/{captions_crossfit_fixed.json, captions_test2015_fixed.json}
set -euo pipefail
OUT=/kaggle/working/fixed_captions
find1() { find -L /kaggle/input -name "$1" -type f 2>/dev/null | head -1; }
TRQ=$(find1 v2_OpenEnded_mscoco_train2014_questions.json)
VAQ=$(find1 v2_OpenEnded_mscoco_val2014_questions.json)
TEQ=$(find1 v2_OpenEnded_mscoco_test2015_questions.json)
for x in "$TRQ" "$VAQ" "$TEQ"; do [ -n "$x" ] || { echo "question file not found under /kaggle/input"; exit 1; }; done
python scripts/fixed_captions.py --questions "$TRQ" "$VAQ" --out "$OUT/captions_crossfit_fixed.json"
python scripts/fixed_captions.py --questions "$TEQ" --out "$OUT/captions_test2015_fixed.json"
ls -la "$OUT"
echo "FIXED_CAPTIONS_DONE"
