#!/usr/bin/env bash
# Step 2 on Kaggle (2x T4): COCO pretrain → fine-tune on the original QD captions.
# Run from QuestionDependentImageCaptioner/:  bash scripts/kaggle_step2.sh
# Resumable: region cache skips existing files, train --continue uses save_dir/last.pt.
set -euo pipefail

PRE=configs/kaggle_coco_pretrain.yaml
FT_BASE=configs/kaggle_mini_ft_base.yaml
COCO=/kaggle/input/datasets/mnhlfuch/vqa-v2/Caption/Caption/captions_train2014.json
QD=/kaggle/input/datasets/mahdirouzmeh/questiondependentcaptions/vqa_v2_question_dependent_captions_train2014.json
PRE_JSON=/kaggle/working/coco_pretrain_captions_train2014.json

echo "== 0/3 build COCO pretrain JSON (QD val/test images excluded) =="
python scripts/build_coco_pretrain.py --coco "$COCO" --qd "$QD" --out "$PRE_JSON"
# Finglish: row range haye config bayad ba info-e JSON-e sakhte shode yeki bashe
python - "$PRE_JSON" "$PRE" <<'EOF'
import json, sys, yaml
info = json.load(open(sys.argv[1]))["info"]
cfg = yaml.safe_load(open(sys.argv[2]))
for k in ("train_row_range", "val_row_range"):
    assert info[k] == cfg[k], f"{k}: JSON {info[k]} != config {cfg[k]} (QD file on Kaggle differs?)"
print("row ranges OK")
EOF

echo "== 1/3 region cache: COCO pretrain images (2 GPUs, ~82k images) =="
python scripts/cache_regions.py --config $PRE --shard 0/2 --device cuda:0 --num-workers 2 --batch-size 16 &
P0=$!
python scripts/cache_regions.py --config $PRE --shard 1/2 --device cuda:1 --num-workers 2 --batch-size 16 &
P1=$!
# Finglish: wait har process joda → age yeki fail kone script stop mishe
wait $P0
wait $P1
# Finglish: image haye val/test-e QD dar pretrain JSON nistan → inja cache mishan
python scripts/cache_regions.py --config $FT_BASE --num-workers 4 --batch-size 16

echo "== 2/3 pretrain on COCO captions =="
torchrun --nproc_per_node=2 train.py --config $PRE --continue

echo "== 3/3 fine-tune on QD captions =="
torchrun --nproc_per_node=2 train.py --config $FT_BASE --continue

# Finglish: faghat best.pt ha baraye download (region cache ~11 GB lazem nist)
mkdir -p /kaggle/working/step2_ckpts
cp outputs/kaggle_coco_pretrain/best.pt /kaggle/working/step2_ckpts/coco_pretrain_best.pt
cp outputs/kaggle_mini_ft_base/best.pt /kaggle/working/step2_ckpts/ft_base_best.pt
echo STEP2_DONE
