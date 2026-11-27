#!/usr/bin/env bash
# Generic-caption control: BLIP-Large as released (Salesforce/blip-image-captioning-large, COCO-caption fine-tuned),
# no training of ours, no question. One caption per image (beam 3), copied to every question of the image.
# Run from the checkout:  cd <repo>/CaptionVQA && PART=trainval bash scripts/kaggle_generic_captions.sh
#   PART=trainval   all train2014 + val2014 images (123,287) -> captions_crossfit_generic.json (658,111 questions)
#   PART=test2015   all test2015 images (81,434)              -> captions_test2015_generic.json (447,793 questions)
# Inputs: mnhlfuch/vqa-v2 (images + questions), a dataset with v2_OpenEnded_mscoco_test2015_questions.json and the
#   test2015 images (PART=test2015). GPU T4 x2, internet on (downloads BLIP-Large, ~1.8 GB).
# Resumable: attach this notebook's own earlier output; finished images are skipped.
# Output: /kaggle/working/generic_captions/
set -euo pipefail
PART=${PART:-trainval}
[[ "$PART" == "trainval" || "$PART" == "test2015" ]] || { echo "PART must be trainval or test2015"; exit 1; }
W=/kaggle/working
OUT=$W/generic_captions
mkdir -p "$OUT"
export PYTHONIOENCODING=utf-8 HF_HOME=$W/hf_cache
START=$SECONDS
read -r NGPU <<<"$(python -c "import torch; print(torch.cuda.device_count())")"
[ "$NGPU" -ge 2 ] || { echo "Need 2 GPUs (found $NGPU): set Settings → Accelerator → GPU T4 x2"; exit 1; }
find1() { find -L /kaggle/input -name "$1" -type f 2>/dev/null | head -1; }

if [ "$PART" = trainval ]; then
  Q=("$(find1 v2_OpenEnded_mscoco_train2014_questions.json)" "$(find1 v2_OpenEnded_mscoco_val2014_questions.json)")
  D=("train2014=$(dirname "$(find1 COCO_train2014_000000000009.jpg)")" "val2014=$(dirname "$(find1 COCO_val2014_000000000042.jpg)")")
else
  Q=("$(find1 v2_OpenEnded_mscoco_test2015_questions.json)")
  D=("test2015=$(dirname "$(find1 COCO_test2015_000000000001.jpg)")")
fi
for x in "${Q[@]}"; do [ -n "$x" ] || { echo "question file not found under /kaggle/input"; exit 1; }; done
for x in "${D[@]}"; do [ "${x#*=}" != "." ] || { echo "images not found: ${x%%=*}"; exit 1; }; done
echo "part $PART | questions: ${Q[*]} | images: ${D[*]}"

# Finglish: resume az output-e notebook-e ghabli
for k in 0 1; do
  f=generic_${PART}_images_shard$k.json
  prev=$(find -L /kaggle/input -name "$f" -type f 2>/dev/null | head -1 || true)
  [ -n "$prev" ] && [ ! -f "$OUT/$f" ] && cp "$prev" "$OUT/" && echo "resume $f"
done

echo "== 1. caption the images (shard k on GPU k) =="
for k in 0 1; do
  python scripts/generic_captions.py caption --part "$PART" --questions "${Q[@]}" --images-dir "${D[@]}" \
    --out-dir "$OUT" --shard $k --num-shards 2 --device cuda:$k > "$OUT/generic_${PART}_gpu$k.log" 2>&1 &
  eval "P$k=\$!"
done
wait $P0; wait $P1
tail -n 3 "$OUT"/generic_${PART}_gpu*.log

echo "== 2. one caption per question + checks =="
python scripts/generic_captions.py expand --part "$PART" --questions "${Q[@]}" --out-dir "$OUT"
rm -rf "$HF_HOME"
ls -la "$OUT"
echo "total $(( (SECONDS - START) / 60 )) min"
echo "GENERIC_CAPTIONS_DONE ($PART)"
