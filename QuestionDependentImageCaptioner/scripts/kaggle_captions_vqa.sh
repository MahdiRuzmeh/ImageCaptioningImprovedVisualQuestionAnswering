#!/usr/bin/env bash
# BLIP captions for the CaptionVQA questions (45,953 = VQA train 30,952 + dev 5,000 + test 10,001) on Kaggle 2× T4.
# Run from the checkout:  cd <repo>/QuestionDependentImageCaptioner && bash scripts/kaggle_captions_vqa.sh
#
# Inputs:
#   - COCO train2014 + val2014 images (mnhlfuch/vqa-v2; paths from configs/b_blip_full.yaml, val2014 found next to it)
#   - the output of the BLIP training notebook (Add Input → Notebooks), holding model_b/b_blip_full/best.pt
# Internet on (BLIP base weights). ~20 min (2 GPUs in parallel, beam 3).
# Output: /kaggle/working/captions_vqa/captions_blip.json  → bring this file back.
set -euo pipefail
W=/kaggle/working
OUT=$W/captions_vqa
mkdir -p "$OUT" "$W/configs"
export PYTHONIOENCODING=utf-8
export HF_HOME=$W/hf_cache

CKPT=$(find -L /kaggle/input -path '*b_blip_full/best.pt' 2>/dev/null | head -1 || true)
[ -z "$CKPT" ] && { echo "best.pt not found: attach the BLIP training notebook output (model_b/b_blip_full/best.pt)"; exit 1; }
TRAIN_IMG=$(python -c "import yaml; print(yaml.safe_load(open('configs/b_blip_full.yaml'))['train_images_dir'])")
if [ ! -f "$TRAIN_IMG/COCO_train2014_000000000009.jpg" ]; then
  TRAIN_IMG=$(dirname "$(find -L /kaggle/input -name COCO_train2014_000000000009.jpg -type f 2>/dev/null | head -1)")
fi
VAL_IMG="$(dirname "$TRAIN_IMG")/val2014"
if [ ! -f "$VAL_IMG/COCO_val2014_000000000042.jpg" ]; then
  VAL_IMG=$(dirname "$(find -L /kaggle/input -name COCO_val2014_000000000042.jpg -type f 2>/dev/null | head -1)")
fi
[ "$TRAIN_IMG" = "." ] || [ "$VAL_IMG" = "." ] && { echo "COCO train2014/val2014 images not found under /kaggle/input"; exit 1; }
echo "ckpt      : $CKPT"
echo "train2014 : $TRAIN_IMG"
echo "val2014   : $VAL_IMG"

python - "$TRAIN_IMG" "$W/configs/b_captions.yaml" <<'EOF'
import sys, yaml
img, dst = sys.argv[1:]
cfg = yaml.safe_load(open("configs/b_blip_full.yaml"))
cfg.update(train_images_dir=img, ddp=False, num_workers=2, eval_batch_size=96)
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
EOF

REQ=../CaptionVQA/data/blip_caption_requests.json
for k in 0 1; do
  python scripts/blip_caption_vqa.py --config "$W/configs/b_captions.yaml" --ckpt "$CKPT" --requests "$REQ" \
    --val-images-dir "$VAL_IMG" --shard $k/2 --device cuda:$k --out "$OUT/captions_shard$k.json" \
    > "$OUT/shard$k.log" 2>&1 &
  eval "P$k=\$!"
done
wait $P0; wait $P1
tail -n 7 "$OUT/shard0.log" "$OUT/shard1.log"
python scripts/blip_caption_vqa.py --merge "$OUT/captions_shard0.json" "$OUT/captions_shard1.json" \
  --out "$OUT/captions_blip.json"
rm -rf "$HF_HOME"
echo CAPTIONS_VQA_DONE
