#!/bin/bash
# Visual-feature comparison: train one variant on QD (seeds 42/43/44), then probe the test split.
# Run from QuestionDependentImageCaptioner/:  bash scripts/run_vis.sh v0_frcnn_box configs/vis_v0_frcnn_box.yaml
# Resumable: a seed with best.pt / probe JSON already there is skipped.
set -euo pipefail
NAME=$1
CFG=$2
PY=../.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
for seed in 42 43 44; do
  DIR=outputs/vis/$NAME/s$seed
  mkdir -p "$DIR"
  if [ ! -f "$DIR/best.pt" ]; then
    echo "=== train $NAME seed $seed ==="
    $PY train.py --config "$CFG" --seed $seed --save-dir "$DIR" --fresh 2>&1 | grep -v Warning > "$DIR.train.log" || { tail -20 "$DIR.train.log"; exit 1; }
    tail -3 "$DIR.train.log"
  fi
  OUT=outputs/vis/$NAME/probe_test_s$seed.json
  if [ ! -f "$OUT" ]; then
    echo "=== probe $NAME seed $seed ==="
    $PY scripts/captioner_probe.py --config "$CFG" --ckpt "$DIR/best.pt" --split test --out "$OUT" 2>&1 \
      | grep -E "^(normal|shuffle|mask) "
  fi
done
echo "VIS_DONE $NAME"
