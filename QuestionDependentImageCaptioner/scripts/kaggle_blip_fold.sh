#!/usr/bin/env bash
# Cross-fitting: fine-tune BLIP on ONE half of the train2014 images (2× T4, torchrun). Two notebooks: FOLD=0 and FOLD=1.
# Run from the checkout:  cd <repo>/QuestionDependentImageCaptioner && FOLD=0 bash scripts/kaggle_blip_fold.sh
#
# Same model and settings as b_blip_full (configs/b_blip_full.yaml): frozen ViT, question as prompt, 4 epochs.
# Inputs: questiondependentcaptions, mnhlfuch/vqa-v2 (train2014 images). Internet on (BLIP base weights).
# ~2–2.5 h. Resumable: attach this notebook's earlier output → it continues from last.pt.
# Output: /kaggle/working/model_b/b_blip_fold<FOLD>/best.pt (+ train.log) → used by CaptionVQA/scripts/kaggle_vqa_full.sh
set -euo pipefail
FOLD=${FOLD:?set FOLD=0 or FOLD=1}
W=/kaggle/working
RUN=b_blip_fold$FOLD
OUT=$W/model_b/$RUN
mkdir -p "$OUT" "$W/configs"
export PYTHONIOENCODING=utf-8
export HF_HOME=$W/hf_cache

QD=$(python scripts/kaggle_find_qd.py configs/b_blip_full.yaml)
IMG=$(python -c "import yaml; print(yaml.safe_load(open('configs/b_blip_full.yaml'))['train_images_dir'])")
if [ ! -f "$IMG/COCO_train2014_000000000009.jpg" ]; then
  IMG=$(dirname "$(find -L /kaggle/input -name COCO_train2014_000000000009.jpg -type f 2>/dev/null | head -1)")
  [ "$IMG" = "." ] && { echo "COCO train2014 images not found under /kaggle/input"; exit 1; }
fi
echo "fold $FOLD | QD $QD | images $IMG"
python -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

prev=$(find -L /kaggle/input -path "*model_b/$RUN/last.pt" 2>/dev/null | head -1 || true)
if [ -n "$prev" ] && [ ! -f "$OUT/last.pt" ]; then echo "resume from $prev"; cp "$(dirname "$prev")"/*.pt "$OUT"/; fi

python scripts/make_fold_qd.py --qd "$QD" --folds ../CaptionVQA/data/crossfit_folds.json --fold "$FOLD" \
  --out-json "$W/qd_fold$FOLD.json" --out-config "$W/configs/$RUN.yaml" --save-dir "$OUT"
python - "$W/configs/$RUN.yaml" "$IMG" <<'EOF'
import sys, yaml
p, img = sys.argv[1:]
cfg = yaml.safe_load(open(p)); cfg["train_images_dir"] = img
yaml.safe_dump(cfg, open(p, "w"), sort_keys=False)
EOF

torchrun --nproc_per_node=2 train_blip.py --config "$W/configs/$RUN.yaml" --continue 2>&1 \
  | stdbuf -oL tr '\r' '\n' | grep --line-buffered -v -E 'it/s\]|s/it\]|it/s, |s/it, |^\s*$' | tee -a "$OUT/train.log"
rm -f "$W/qd_fold$FOLD.json"
rm -rf "$HF_HOME"
ls -la "$OUT"
echo "BLIP_FOLD${FOLD}_DONE"
