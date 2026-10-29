#!/usr/bin/env bash
# Step 2 on Kaggle (2x T4): COCO pretrain → fine-tune on enriched QD + original QD (ablation).
# Run from QuestionDependentImageCaptioner/:  bash scripts/kaggle_step2.sh
# Resumable: region cache skips existing files, train --continue uses save_dir/last.pt.
set -euo pipefail

PRE=configs/kaggle_coco_pretrain.yaml
FT_ENR=configs/kaggle_mini_ft_enriched.yaml
FT_BASE=configs/kaggle_mini_ft_base.yaml

echo "== 1/4 region cache: COCO pretrain images (2 GPUs, ~82k images) =="
python scripts/cache_regions.py --config $PRE --shard 0/2 --device cuda:0 --num-workers 2 --batch-size 16 &
python scripts/cache_regions.py --config $PRE --shard 1/2 --device cuda:1 --num-workers 2 --batch-size 16 &
wait
# Finglish: image haye val/test-e QD dar pretrain JSON nistan → inja cache mishan
python scripts/cache_regions.py --config $FT_ENR --num-workers 4 --batch-size 16

echo "== 2/4 pretrain on COCO captions =="
torchrun --nproc_per_node=2 train.py --config $PRE --continue

echo "== 3/4 fine-tune on enriched QD captions =="
torchrun --nproc_per_node=2 train.py --config $FT_ENR --continue

echo "== 4/4 fine-tune on original QD captions (ablation) =="
torchrun --nproc_per_node=2 train.py --config $FT_BASE --continue

# Finglish: faghat best.pt ha baraye download (region cache ~11 GB lazem nist)
mkdir -p /kaggle/working/step2_ckpts
cp outputs/kaggle_coco_pretrain/best.pt /kaggle/working/step2_ckpts/coco_pretrain_best.pt
cp outputs/kaggle_mini_ft_enriched/best.pt /kaggle/working/step2_ckpts/ft_enriched_best.pt
cp outputs/kaggle_mini_ft_base/best.pt /kaggle/working/step2_ckpts/ft_base_best.pt
echo STEP2_DONE
