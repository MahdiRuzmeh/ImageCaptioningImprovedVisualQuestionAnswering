#!/usr/bin/env bash
# Grounding test for model B (BLIP) on Kaggle: normal / no_image / no_question / swap_image / swap_question.
# Run from the checkout:  cd <repo>/QuestionDependentImageCaptioner && bash scripts/kaggle_grounding_b.sh
#
# Inputs:
#   - QD captions + COCO train2014 at the paths in configs/b_blip_full.yaml
#   - the output of the BLIP training notebook (Add Input → Notebooks), holding model_b/b_blip_full/best.pt
# Internet on (BLIP base weights + NLTK WordNet). One T4 is enough (~40 min: 5 × 10k captions, beam 3).
# Output: /kaggle/working/grounding_b/grounding.json  → bring this file back.
set -euo pipefail
W=/kaggle/working
OUT=$W/grounding_b
mkdir -p "$OUT" "$W/configs"
export PYTHONIOENCODING=utf-8
export HF_HOME=$W/hf_cache

CKPT=$(find -L /kaggle/input -path '*b_blip_full/best.pt' 2>/dev/null | head -1 || true)
[ -z "$CKPT" ] && { echo "best.pt not found: attach the BLIP training notebook output (model_b/b_blip_full/best.pt)"; exit 1; }
QD=$(python scripts/kaggle_find_qd.py configs/b_blip_full.yaml)
IMG=$(python -c "import yaml; print(yaml.safe_load(open('configs/b_blip_full.yaml'))['train_images_dir'])")
if [ ! -f "$IMG/COCO_train2014_000000000009.jpg" ]; then
  IMG=$(dirname "$(find -L /kaggle/input -name COCO_train2014_000000000009.jpg -type f 2>/dev/null | head -1)")
  [ "$IMG" = "." ] && { echo "COCO train2014 images not found under /kaggle/input"; exit 1; }
fi
echo "ckpt    : $CKPT"
echo "QD json : $QD"
echo "images  : $IMG"
python -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

python - "$QD" "$IMG" "$OUT" "$W/configs/b_grounding.yaml" <<'EOF'
import sys, yaml
qd, img, save, dst = sys.argv[1:]
cfg = yaml.safe_load(open("configs/b_blip_full.yaml"))
cfg.update(train_captions_json=qd, val_captions_json=qd, train_images_dir=img, val_images_dir=img, save_dir=save,
           ddp=False, num_workers=4)
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
EOF

python scripts/grounding_eval.py --family blip --config "$W/configs/b_grounding.yaml" --ckpt "$CKPT" \
  --name b_blip_full --out "$OUT/grounding.json" 2>&1 | stdbuf -oL tr '\r' '\n' | grep -v -E 'it/s\]|s/it\]|^\s*$'
rm -rf "$HF_HOME"
echo GROUNDING_B_DONE
