#!/usr/bin/env bash
# BUTD features for ALL 81,434 test2015 images (VQA v2 test-dev + test-standard) → /kaggle/working/butd_test2015/
# (~12 GB, packed). Then: notebook Output → New Dataset (e.g. "qd-butd-test2015"); the FINAL VQA notebook reads it.
# Run from the checkout:  cd <repo>/CaptionVQA && bash scripts/kaggle_test2015_cache.sh
# Inputs: mnhlfuch/vqa-v2. GPU T4 x2, internet on (BUTD weights). ~4–5 h (2 GPUs). Resumable (attach own output).
set -euo pipefail
W=/kaggle/working
OUT=$W/butd_test2015
CAP_ROOT=../QuestionDependentImageCaptioner
export PYTHONIOENCODING=utf-8
read -r NGPU <<<"$(python -c "import torch; print(torch.cuda.device_count())")"
[ "$NGPU" -ge 1 ] || { echo "No GPU: set Settings → Accelerator → GPU T4 x2"; exit 1; }
TEIMG=$(dirname "$(find -L /kaggle/input -name COCO_test2015_000000000001.jpg -type f 2>/dev/null | head -1)")
[ "$TEIMG" != "." ] || { echo "test2015 images not found (attach mnhlfuch/vqa-v2)"; exit 1; }
echo "test2015: $TEIMG | GPUs: $NGPU"

BUTD_URL=https://s3.amazonaws.com/models.huggingface.co/bert/unc-nlp/frcnn-vg-finetuned
BUTD_SHA=d3bf8ad74828a244a95d85e17a842cc0ac75419a53e4e81ee68386f192bb24c5
WD=$CAP_ROOT/outputs/butd_weights
if [ ! -f "$WD/pytorch_model.bin" ]; then
  mkdir -p "$WD"
  curl -sfL --retry 3 -o "$WD/config.yaml" "$BUTD_URL/config.yaml"
  curl -sfL --retry 3 -o "$WD/pytorch_model.bin" "$BUTD_URL/pytorch_model.bin"
  echo "$BUTD_SHA  $WD/pytorch_model.bin" | sha256sum -c --quiet && echo "BUTD weights OK (sha256)"
fi
# Finglish: resume — age khorooji-e session-e ghabli attach shode, copy kon (done flag ha mimoonan)
PREV=$(find -L /kaggle/input -path '*butd_test2015/meta.json' 2>/dev/null | head -1 || true)
if [ -n "$PREV" ] && [ ! -f "$OUT/meta.json" ]; then cp -r "$(dirname "$PREV")" "$OUT"; echo "resume from $PREV"; fi

python extract_images_packed.py --images-dir "$TEIMG" --prefix COCO_test2015_ --out "$OUT" --init-only
for k in $(seq 0 $((NGPU - 1))); do
  python extract_images_packed.py --images-dir "$TEIMG" --prefix COCO_test2015_ --out "$OUT" \
    --shard $k/$NGPU --device cuda:$k > "$W/test2015_gpu$k.log" 2>&1 &
done
wait
grep -h "shard" "$W"/test2015_gpu*.log | grep -v "it/s" || true
python - "$OUT" <<'EOF'
import json, sys, numpy as np
d = sys.argv[1]; n = len(json.load(open(f"{d}/meta.json"))["ids"])
done = int(np.memmap(f"{d}/done.u8", np.uint8, "r", shape=(n,)).sum())
print(f"test2015 cache: {done}/{n} images extracted")
sys.exit(0 if done == n else 1)
EOF
echo TEST2015_CACHE_DONE
