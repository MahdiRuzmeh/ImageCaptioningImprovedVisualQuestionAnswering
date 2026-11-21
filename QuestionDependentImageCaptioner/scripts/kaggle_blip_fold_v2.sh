#!/usr/bin/env bash
# Round 2: fold captioner with the top 4 ViT layers trained (lr x0.1) + 6 epochs. Same data/fold as kaggle_blip_fold.sh,
# config configs/b_blip_fold_v2.yaml.
# Other fold configs: CONFIG=configs/b_blip_fold_large.yaml NAME=large FOLD=0 bash scripts/kaggle_blip_fold_v2.sh
# Then scores the new captioner on 10,000 QD rows of the OTHER fold (images it never saw), beam 3.
#   cd <repo>/QuestionDependentImageCaptioner && FOLD=0 bash scripts/kaggle_blip_fold_v2.sh
# Inputs: questiondependentcaptions, mnhlfuch/vqa-v2. Internet on. GPU T4 x2.
# Resumable: attach this notebook's earlier output → continues from last.pt.
# Output: /kaggle/working/model_b/b_blip_fold<FOLD>_v2/{best.pt,train.log,fold_eval.json}
set -euo pipefail
FOLD=${FOLD:?set FOLD=0 or FOLD=1}
EVAL_N=${EVAL_N:-10000}
W=/kaggle/working
CONFIG=${CONFIG:-configs/b_blip_fold_v2.yaml}
NAME=${NAME:-v2}
RUN=b_blip_fold${FOLD}_${NAME}
OUT=$W/model_b/$RUN
mkdir -p "$OUT" "$W/configs"
export PYTHONIOENCODING=utf-8
export HF_HOME=$W/hf_cache

read -r NGPU NCCL TV <<<"$(python -c "import torch, torch.distributed as d; print(torch.cuda.device_count(), d.is_nccl_available(), torch.__version__)")"
echo "torch $TV | GPUs: $NGPU | NCCL available: $NCCL"
if [ "$NGPU" -lt 1 ]; then
  echo "No GPU in this session: set Settings → Accelerator → GPU T4 x2, then run again."; exit 1
fi
[ "$NGPU" -lt 2 ] && echo "WARNING: only $NGPU GPU → training on 1 GPU (about 2× slower)"

QD=$(python scripts/kaggle_find_qd.py configs/b_blip_full.yaml)
IMG=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['train_images_dir'])")
if [ ! -f "$IMG/COCO_train2014_000000000009.jpg" ]; then
  IMG=$(dirname "$(find -L /kaggle/input -name COCO_train2014_000000000009.jpg -type f 2>/dev/null | head -1)")
  [ "$IMG" = "." ] && { echo "COCO train2014 images not found under /kaggle/input"; exit 1; }
fi
echo "fold $FOLD | QD $QD | images $IMG"
python -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

prev=$(find -L /kaggle/input -path "*model_b/$RUN/last.pt" 2>/dev/null | head -1 || true)
if [ -n "$prev" ] && [ ! -f "$OUT/last.pt" ]; then echo "resume from $prev"; cp "$(dirname "$prev")"/*.pt "$OUT"/; fi

python scripts/make_fold_qd.py --qd "$QD" --folds ../CaptionVQA/data/crossfit_folds.json --fold "$FOLD" \
  --base-config "$CONFIG" \
  --out-json "$W/qd_fold$FOLD.json" --out-config "$W/configs/$RUN.yaml" --save-dir "$OUT"
python - "$W/configs/$RUN.yaml" "$IMG" <<'PYEOF'
import sys, yaml
p, img = sys.argv[1:]
cfg = yaml.safe_load(open(p)); cfg["train_images_dir"] = img
yaml.safe_dump(cfg, open(p, "w"), sort_keys=False)
print(f"config: {cfg['blip_model']} unfreeze_vision_layers={cfg.get('unfreeze_vision_layers', 0)} batch={cfg['batch_size']} "
      f"epochs={cfg['epochs']}")
PYEOF

FILTER='it/s\]|s/it\]|it/s, |s/it, |^\s*$'
torchrun --nproc_per_node=$NGPU train_blip.py --config "$W/configs/$RUN.yaml" --continue 2>&1 \
  | stdbuf -oL tr '\r' '\n' | grep --line-buffered -v -E "$FILTER" | tee -a "$OUT/train.log"

if python -c "import torch,sys; s=torch.load('$OUT/last.pt',map_location='cpu',weights_only=False); c=s['config']; sys.exit(0 if s['epoch']>=int(c['epochs']) else 1)"; then
  echo "training complete → scoring on $EVAL_N rows of fold $((1 - FOLD))"
  torchrun --nproc_per_node=$NGPU scripts/eval_fold_captioner.py --qd "$QD" \
    --folds ../CaptionVQA/data/crossfit_folds.json --fold "$FOLD" --n "$EVAL_N" \
    --config "$W/configs/$RUN.yaml" --ckpt "$OUT/best.pt" --name "$RUN" --out "$OUT/fold_eval.json" 2>&1 \
    | stdbuf -oL tr '\r' '\n' | grep --line-buffered -v -E "$FILTER" | tee "$OUT/fold_eval.log"
  rm -f "$OUT/last.pt"   # Finglish: kamel shod → last.pt (optimizer, ~2x size) lazem nist
else
  echo "training not finished (time budget) → run this notebook again with its output attached (resumes)"
fi
rm -f "$W/qd_fold$FOLD.json"
rm -rf "$HF_HOME"
ls -la "$OUT"
echo "BLIP_FOLD${FOLD}_${NAME^^}_DONE"
