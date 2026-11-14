#!/usr/bin/env bash
# Model A on Kaggle (2x T4): BUTD packed cache → train a_full_butd (+ a_full_butd_large) → test metrics.
# Run from the checkout:  cd <repo>/QuestionDependentImageCaptioner && bash scripts/kaggle_model_a.sh
#
# Inputs (attach as Kaggle datasets):
#   - QD captions + COCO train2014 JPEGs at the paths in configs/a_full_butd.yaml
#     (/kaggle/input/questiondependentcaptions/..., /kaggle/input/datasets/mnhlfuch/vqa-v2/...).
#     A missing path is searched by file name under /kaggle/input; the QD file is checked by row count.
#   - BUTD weights: downloaded from HF's legacy S3 bucket (internet on, sha256 checked); only if that fails,
#     a dataset holding config.yaml + pytorch_model.bin is used
#   - optional: an earlier output of this notebook → its butd_packed/ (skip extraction) and
#     model_a/<run>/last.pt (resume training)
#
# Env:
#   STAGES="extract train_a train_a_large"   (default: all three; drop what you do not need)
#   EPOCHS=15                                  override epochs for both runs
# Everything is resumable: extraction skips finished images, training uses --continue (last.pt per epoch).
# Output (/kaggle/working): butd_packed/ (~12 GB, reuse next session), model_a/<run>/{best.pt,last.pt,test_metrics.json,train.log},
# model_a/results.md
set -euo pipefail

STAGES=${STAGES:-"extract train_a train_a_large"}
W=/kaggle/working
PACKED=$W/butd_packed
OUT=$W/model_a
CFG_DIR=$W/configs
mkdir -p "$OUT" "$CFG_DIR"
export PYTHONIOENCODING=utf-8

find_one() {  # find_one <name> [type] → first match under /kaggle/input
  find -L /kaggle/input -name "$1" ${2:+-type $2} 2>/dev/null | head -1
}

# Finglish: file-e QD ba row count-e split check mishe (JSON-e ghadimi ghabool nemishe)
QD=$(python scripts/kaggle_find_qd.py configs/a_full_butd.yaml)
# Finglish: image dir az config; age nabashe search
IMG=$(python -c "import yaml; print(yaml.safe_load(open('configs/a_full_butd.yaml'))['train_images_dir'])")
if [ ! -f "$IMG/COCO_train2014_000000000009.jpg" ]; then
  echo "config train_images_dir not found ($IMG), searching /kaggle/input"
  IMG=$(dirname "$(find_one COCO_train2014_000000000009.jpg f)")
  [ "$IMG" = "." ] && { echo "COCO train2014 images not found under /kaggle/input"; exit 1; }
fi
echo "QD json : $QD"
echo "images  : $IMG"

echo "== 0. setup: NLTK WordNet (METEOR), BUTD weights =="
python -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"
# Finglish: BUTD weight — repo-e HF (unc-nlp/frcnn-vg-finetuned) dige baz nemishe, vali bucket-e S3-e
# ghadimi-e HF hamoon file ro dare (sha256 = hamoon file-e PC, check shode 2026-11-14).
# Download fail beshe → fallback: dataset-e attach shode ba config.yaml + pytorch_model.bin
BUTD_URL=https://s3.amazonaws.com/models.huggingface.co/bert/unc-nlp/frcnn-vg-finetuned
BUTD_SHA=d3bf8ad74828a244a95d85e17a842cc0ac75419a53e4e81ee68386f192bb24c5
if [ ! -f outputs/butd_weights/pytorch_model.bin ]; then
  mkdir -p outputs/butd_weights
  if curl -sfL --retry 3 -o outputs/butd_weights/config.yaml "$BUTD_URL/config.yaml" \
     && curl -sfL --retry 3 -o outputs/butd_weights/pytorch_model.bin "$BUTD_URL/pytorch_model.bin" \
     && echo "$BUTD_SHA  outputs/butd_weights/pytorch_model.bin" | sha256sum -c --quiet; then
    echo "BUTD weights downloaded from $BUTD_URL (sha256 OK)"
  else
    echo "BUTD download failed or checksum mismatch; looking for an attached dataset"
    rm -f outputs/butd_weights/config.yaml outputs/butd_weights/pytorch_model.bin
  fi
fi
if [ ! -f outputs/butd_weights/pytorch_model.bin ]; then
  WDIR=""
  while IFS= read -r c; do
    d=$(dirname "$c")
    if [ -f "$d/pytorch_model.bin" ] && grep -q "proposal_generator" "$c"; then WDIR=$d; break; fi
  done < <(find -L /kaggle/input -name config.yaml -type f 2>/dev/null)
  if [ -z "$WDIR" ]; then
    echo "BUTD weights not found: attach a dataset with config.yaml + pytorch_model.bin"
    echo "(outputs/butd_weights on the PC; the Hugging Face repo unc-nlp/frcnn-vg-finetuned is gone)"
    exit 1
  fi
  echo "BUTD weights: $WDIR"
  mkdir -p outputs/butd_weights && cp "$WDIR/config.yaml" "$WDIR/pytorch_model.bin" outputs/butd_weights/
fi

# Earlier session output attached? → reuse its packed cache / checkpoints (copy into the writable working dir)
PREV_META=$(find -L /kaggle/input -path '*butd_packed/meta.json' 2>/dev/null | head -1 || true)
if [ -n "$PREV_META" ] && [ ! -f "$PACKED/meta.json" ]; then
  if [[ " $STAGES " == *" extract "* ]]; then
    # Finglish: extract bayad benevise → copy be working (resume-e extraction)
    echo "copying packed cache from $(dirname "$PREV_META") (extraction resumes)"
    mkdir -p "$PACKED" && cp -r "$(dirname "$PREV_META")"/. "$PACKED"/
  else
    # Finglish: faghat train → memmap read-only az input (bedoon copy-e 12 GB, output kochik)
    PACKED=$(dirname "$PREV_META")
    echo "using packed cache in place (read-only): $PACKED"
  fi
fi
if [[ " $STAGES " != *" extract "* ]] && [ ! -f "$PACKED/meta.json" ]; then
  echo "No BUTD packed cache: attach the output of the STAGES=extract run (it holds butd_packed/meta.json)"; exit 1
fi
if [[ " $STAGES " != *" extract "* ]]; then
  python - "$PACKED" <<'EOF'
import sys, numpy as np, json, pathlib
d = pathlib.Path(sys.argv[1]); n = len(json.loads((d / "meta.json").read_text())["ids"])
done = int(np.memmap(d / "done.u8", np.uint8, "r", shape=(n,)).sum())
print(f"packed cache: {done}/{n} images extracted")
sys.exit(0 if done == n else "extraction incomplete: rerun STAGES=extract with this output attached")
EOF
fi
for run in a_full_butd a_full_butd_large; do
  prev=$(find -L /kaggle/input -path "*model_a/$run/last.pt" 2>/dev/null | head -1 || true)
  if [ -n "$prev" ] && [ ! -f "$OUT/$run/last.pt" ]; then
    echo "resume $run from $prev"; mkdir -p "$OUT/$run" && cp "$(dirname "$prev")"/*.pt "$OUT/$run"/
  fi
done

# Runtime configs: same YAML, Kaggle paths
for run in a_full_butd a_full_butd_large; do
  python - "$run" "$QD" "$IMG" "$PACKED" "$OUT/$run" "$CFG_DIR/$run.yaml" "${EPOCHS:-}" <<'EOF'
import sys, yaml
run, qd, img, packed, save, dst, epochs = sys.argv[1:]
cfg = yaml.safe_load(open(f"configs/{run}.yaml"))
cfg.update(train_captions_json=qd, val_captions_json=qd, train_images_dir=img, val_images_dir=img,
           train_region_cache_dir=packed, val_region_cache_dir=packed, test_region_cache_dir=packed, save_dir=save)
if epochs:
    cfg["epochs"] = int(epochs)
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
print("config", dst)
EOF
done

if [[ " $STAGES " == *" extract "* ]]; then
  echo "== 1. BUTD packed cache (2 GPUs) =="
  python scripts/extract_visual.py --config "$CFG_DIR/a_full_butd.yaml" --packed-dir "$PACKED" --init-only
  python scripts/extract_visual.py --config "$CFG_DIR/a_full_butd.yaml" --packed-dir "$PACKED" \
    --shard 0/2 --device cuda:0 --batch-size 1 --num-workers 2 > "$W/extract_gpu0.log" 2>&1 &
  P0=$!
  python scripts/extract_visual.py --config "$CFG_DIR/a_full_butd.yaml" --packed-dir "$PACKED" \
    --shard 1/2 --device cuda:1 --batch-size 1 --num-workers 2 > "$W/extract_gpu1.log" 2>&1 &
  P1=$!
  wait $P0; wait $P1
  tail -n 2 "$W/extract_gpu0.log" "$W/extract_gpu1.log"
fi

train_and_test() {  # train_and_test <run>
  local run=$1
  echo "== train $run =="
  # Finglish: tqdm \r → \n, bad progress line ha hazf (epoch summary mimune)
  torchrun --nproc_per_node=2 train.py --config "$CFG_DIR/$run.yaml" --continue 2>&1 \
    | stdbuf -oL tr '\r' '\n' | grep --line-buffered -v -E 'it/s\]|it/s, |^\s*$' | tee -a "$OUT/$run/train.log"
  echo "== test $run (beam 3) =="
  python benchmark.py --config "$CFG_DIR/$run.yaml" --ckpt "$OUT/$run/best.pt" --name "$run" --beam 3 \
    2>&1 | tee "$OUT/$run/test.log" | tail -n 25
}
mkdir -p "$OUT/a_full_butd" "$OUT/a_full_butd_large"
[[ " $STAGES " == *" train_a "* ]] && train_and_test a_full_butd
[[ " $STAGES " == *" train_a_large "* ]] && train_and_test a_full_butd_large

python scripts/results_table.py --glob "$OUT/*/test_metrics.json" --out "$OUT/results.md" || true
echo MODEL_A_DONE
