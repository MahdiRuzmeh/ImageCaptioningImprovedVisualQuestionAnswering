#!/usr/bin/env bash
# Model B on Kaggle (2x T4): BLIP-base question-prompted captioner → train → test metrics.
# Run from the checkout:  cd <repo>/src/QuestionDependentImageCaptioner && bash scripts/kaggle_model_b.sh
#
# Inputs (Kaggle datasets, found by file name under /kaggle/input):
#   - QD caption JSON (full generator file or prepare_full_qd.py output; checked by row count)
#   - COCO train2014 JPEGs (folder holding COCO_train2014_000000000009.jpg)
#   - optional: an earlier output of this notebook → model_b/b_blip_full/last.pt (resume)
# Internet on: BLIP weights (Salesforce/blip-image-captioning-base, ~1 GB) + NLTK WordNet.
#
# Env: STAGES="train test" (default), EPOCHS=<n>, MAX_HOURS=<h> (default 10.5 from the config)
# Output (/kaggle/working/model_b/b_blip_full): best.pt, last.pt, test_metrics.json, train.log; model_b/results.md
set -euo pipefail

STAGES=${STAGES:-"train test"}
W=/kaggle/working
RUN=b_blip_full
OUT=$W/model_b
CFG=$W/configs/$RUN.yaml
mkdir -p "$OUT/$RUN" "$W/configs"
export PYTHONIOENCODING=utf-8
export HF_HOME=$W/hf_cache   # Finglish: BLIP weights; ghabl az tamoom shodan pak mishe (output kochik bemune)

QD=$(python scripts/kaggle_find_qd.py configs/$RUN.yaml)
IMG=$(dirname "$(find -L /kaggle/input -name COCO_train2014_000000000009.jpg -type f 2>/dev/null | head -1)")
[ "$IMG" = "." ] && { echo "COCO train2014 images not found under /kaggle/input"; exit 1; }
echo "QD json : $QD"
echo "images  : $IMG"

python -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

prev=$(find -L /kaggle/input -path "*model_b/$RUN/last.pt" 2>/dev/null | head -1 || true)
if [ -n "$prev" ] && [ ! -f "$OUT/$RUN/last.pt" ]; then
  echo "resume from $prev"; cp "$(dirname "$prev")"/*.pt "$OUT/$RUN"/
fi

python - "$QD" "$IMG" "$OUT/$RUN" "$CFG" "${EPOCHS:-}" "${MAX_HOURS:-}" <<'EOF'
import sys, yaml
qd, img, save, dst, epochs, hours = sys.argv[1:]
cfg = yaml.safe_load(open("configs/b_blip_full.yaml"))
cfg.update(train_captions_json=qd, train_images_dir=img, save_dir=save)
if epochs:
    cfg["epochs"] = int(epochs)
if hours:
    cfg["max_hours"] = float(hours)
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
print("config", dst)
EOF

if [[ " $STAGES " == *" train "* ]]; then
  echo "== train $RUN =="
  # Finglish: tqdm \r → \n, progress line ha hazf (epoch summary + mesal ha mimoonan)
  torchrun --nproc_per_node=2 train_blip.py --config "$CFG" --continue 2>&1 \
    | stdbuf -oL tr '\r' '\n' | grep --line-buffered -v -E 'it/s\]|it/s, |^\s*$' | tee -a "$OUT/$RUN/train.log"
fi
if [[ " $STAGES " == *" test "* ]]; then
  echo "== test $RUN (beam 3) =="
  python train_blip.py --config "$CFG" --test --beam 3 --name "$RUN" 2>&1 \
    | stdbuf -oL tr '\r' '\n' | grep -v -E 'it/s\]|^\s*$' | tee "$OUT/$RUN/test.log" | tail -n 20
fi

python scripts/results_table.py --glob "$OUT/*/test_metrics.json" --out "$OUT/results.md" || true
rm -rf "$HF_HOME"
echo MODEL_B_DONE
