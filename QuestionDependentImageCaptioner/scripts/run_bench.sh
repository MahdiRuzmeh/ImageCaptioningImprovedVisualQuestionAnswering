#!/usr/bin/env bash
# Train + benchmark yek architecture roye chand seed.
#   bash scripts/run_bench.sh configs/bench_v1.yaml v1 [python] [seeds...]
# Run az QuestionDependentImageCaptioner/.
set -euo pipefail
CONFIG="$1"
NAME="$2"
PY="${3:-../.venv/Scripts/python.exe}"
shift $(( $# >= 3 ? 3 : $# ))
SEEDS=("${@:-42 43 44}")
[ "$#" -eq 0 ] && SEEDS=(42 43 44)
export PYTHONIOENCODING=utf-8

for SEED in "${SEEDS[@]}"; do
  OUT="outputs/bench/${NAME}/seed${SEED}"
  echo "=== ${NAME} seed=${SEED} → ${OUT}"
  "$PY" train.py --config "$CONFIG" --seed "$SEED" --save-dir "$OUT" --fresh \
    2>&1 | tr '\r' '\n' | grep -E "^(split=|device=|epoch |  best|Traceback|  File|[A-Za-z]+Error)" || true
  "$PY" benchmark.py --config "$CONFIG" --ckpt "$OUT/best.pt" --name "$NAME" \
    2>&1 | grep -v -i warning | head -12
done
