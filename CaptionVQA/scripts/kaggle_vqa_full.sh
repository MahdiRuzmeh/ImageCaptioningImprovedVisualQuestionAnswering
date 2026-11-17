#!/usr/bin/env bash
# Fair control on Kaggle (2× T4): VQA trained on ALL of train2014 (443,757 questions); same test as before.
# Run from the checkout:  cd <repo>/CaptionVQA && MODE=baseline bash scripts/kaggle_vqa_full.sh
#   MODE=baseline : no captions, seeds 0 1 2                                         (~1.5 h)
#   MODE=crossfit : + cross-fitted BLIP captions (each question captioned by the fold model that never
#                   saw its image), seeds 0 1 2                                        (~4.5 h)
# Both GPUs are used everywhere: region extraction (2 shards), captions (fold-0 model on GPU 0, fold-1 on GPU 1),
# training (2 seeds in parallel).
#
# Inputs (Add Input):
#   - mnhlfuch/vqa-v2, questiondependentcaptions
#   - the BUTD packed cache from model A (the dataset made from the STAGES=extract output; holds butd_packed/meta.json)
#     → 79,568 images are copied instead of re-extracted (without it: +3 h of extraction)
#   - MODE=crossfit: the outputs of the two fold notebooks (QuestionDependentImageCaptioner/scripts/kaggle_blip_fold.sh,
#     FOLD=0 and FOLD=1) holding model_b/b_blip_fold0/best.pt and model_b/b_blip_fold1/best.pt
# Internet on. Resumable: attach this notebook's earlier output (captions_*.json are reused).
# Output: /kaggle/working/vqa_full_<MODE>/ (runs/, summary.md, captions/) → bring this folder back.
set -euo pipefail
MODE=${MODE:-baseline}
[[ "$MODE" == "baseline" || "$MODE" == "crossfit" ]] || { echo "MODE must be baseline or crossfit"; exit 1; }
W=/kaggle/working
WORK=$W/vqa_full_$MODE
CFG=$W/configs/vqa_full_$MODE.yaml
CAP_ROOT=../QuestionDependentImageCaptioner
mkdir -p "$WORK/captions" "$W/configs"
export PYTHONIOENCODING=utf-8

find1() { find -L /kaggle/input -name "$1" -type f 2>/dev/null | head -1; }
TRQ=$(find1 v2_OpenEnded_mscoco_train2014_questions.json); TRA=$(find1 v2_mscoco_train2014_annotations.json)
VAQ=$(find1 v2_OpenEnded_mscoco_val2014_questions.json);   VAA=$(find1 v2_mscoco_val2014_annotations.json)
TRIMG=$(dirname "$(find1 COCO_train2014_000000000009.jpg)"); VAIMG=$(dirname "$(find1 COCO_val2014_000000000042.jpg)")
QD=$(cd "$CAP_ROOT" && python scripts/kaggle_find_qd.py configs/b_blip_full.yaml)
PACKED=$(dirname "$(find -L /kaggle/input -path '*butd_packed/meta.json' 2>/dev/null | head -1)")
for v in TRQ TRA VAQ VAA; do [ -n "${!v}" ] || { echo "VQA v2 file missing ($v)"; exit 1; }; done
[ "$TRIMG" != "." ] && [ "$VAIMG" != "." ] || { echo "COCO train2014/val2014 images not found"; exit 1; }
[ "$PACKED" = "." ] && { echo "WARNING: no butd_packed cache attached -> all 82,783 train images are extracted (+3 h)"; PACKED=""; }
printf 'MODE %s\nQD %s\nimages %s | %s\nBUTD cache %s\n' "$MODE" "$QD" "$TRIMG" "$VAIMG" "${PACKED:-none}"

python - "$CFG" "$WORK" "$QD" "$TRQ" "$TRA" "$VAQ" "$VAA" "$TRIMG" "$VAIMG" <<'EOF'
import sys, yaml
dst, work, qd, trq, tra, vaq, vaa, trimg, vaimg = sys.argv[1:]
cfg = yaml.safe_load(open("configs/blip_full.yaml"))
cfg.update(work_dir=work, qd_captions_json=qd, train_questions_json=trq, train_annotations_json=tra,
           val_questions_json=vaq, val_annotations_json=vaa, train_images_dir=trimg, val_images_dir=vaimg,
           caption_requests_json=f"{work}/caption_requests.json", captions_json=f"{work}/captions/captions_crossfit.json")
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
EOF

echo "== 1. split (all train2014) =="
python make_split.py --config "$CFG"
python - "$WORK" <<'EOF'
import json, sys
work = sys.argv[1]
req = json.load(open(f"{work}/caption_requests.json"))
small = json.load(open("data/blip_caption_requests.json"))
for s in ("dev", "test"):
    a = [r["question_id"] for r in req if r["vqa_split"] == s]
    b = [r["question_id"] for r in small if r["vqa_split"] == s]
    assert a == b, f"{s} differs from the earlier benchmark"
assert all("caption_model" in r for r in req), "caption_model missing (data/crossfit_folds.json)"
print(f"dev/test identical to the earlier benchmark; {len(req)} questions with a fold caption model")
EOF

echo "== 2. BUTD features (copy from cache, then extract the rest on 2 GPUs) =="
BUTD_URL=https://s3.amazonaws.com/models.huggingface.co/bert/unc-nlp/frcnn-vg-finetuned
BUTD_SHA=d3bf8ad74828a244a95d85e17a842cc0ac75419a53e4e81ee68386f192bb24c5
WD=$CAP_ROOT/outputs/butd_weights
if [ ! -f "$WD/pytorch_model.bin" ]; then
  mkdir -p "$WD"
  curl -sfL --retry 3 -o "$WD/config.yaml" "$BUTD_URL/config.yaml"
  curl -sfL --retry 3 -o "$WD/pytorch_model.bin" "$BUTD_URL/pytorch_model.bin"
  echo "$BUTD_SHA  $WD/pytorch_model.bin" | sha256sum -c --quiet && echo "BUTD weights OK (sha256)"
fi
python prepare_regions.py --config "$CFG" --init-only ${PACKED:+--packed-dir "$PACKED"} 2>&1 | grep -v "it/s"
for k in 0 1; do
  python prepare_regions.py --config "$CFG" --shard $k/2 --device cuda:$k > "$WORK/regions_gpu$k.log" 2>&1 &
  eval "R$k=\$!"
done
wait $R0; wait $R1
grep -h "shard" "$WORK"/regions_gpu*.log | grep -v "it/s" || true

CAPS=none
if [ "$MODE" = "crossfit" ]; then
  CAPS=blip
  echo "== 3. cross-fitted BLIP captions (fold-0 model on GPU 0, fold-1 model on GPU 1) =="
  C0=$(find -L /kaggle/input -path '*model_b/b_blip_fold0/best.pt' 2>/dev/null | head -1 || true)
  C1=$(find -L /kaggle/input -path '*model_b/b_blip_fold1/best.pt' 2>/dev/null | head -1 || true)
  [ -n "$C0" ] && [ -n "$C1" ] || { echo "attach both fold notebook outputs (b_blip_fold0/best.pt, b_blip_fold1/best.pt)"; exit 1; }
  echo "fold 0: $C0"; echo "fold 1: $C1"
  # Finglish: resume — caption haye session-e ghabl (age attach shode) copy mishan
  for k in 0 1; do
    prev=$(find -L /kaggle/input -name "captions_model$k.json" 2>/dev/null | head -1 || true)
    [ -n "$prev" ] && [ ! -f "$WORK/captions/captions_model$k.json" ] && cp "$prev" "$WORK/captions/" && echo "resume captions_model$k.json"
  done
  python - "$TRIMG" "$W/configs/b_captions.yaml" <<'EOF'
import sys, yaml
img, dst = sys.argv[1:]
cfg = yaml.safe_load(open("../QuestionDependentImageCaptioner/configs/b_blip_full.yaml"))
cfg.update(train_images_dir=img, ddp=False, num_workers=2, eval_batch_size=96)
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
EOF
  export HF_HOME=$W/hf_cache
  for k in 0 1; do
    CK=$([ $k = 0 ] && echo "$C0" || echo "$C1")
    (cd "$CAP_ROOT" && python scripts/blip_caption_vqa.py --config "$W/configs/b_captions.yaml" --ckpt "$CK" \
      --requests "$WORK/caption_requests.json" --val-images-dir "$VAIMG" --caption-model $k --device cuda:$k \
      --out "$WORK/captions/captions_model$k.json") > "$WORK/captions_gpu$k.log" 2>&1 &
    eval "P$k=\$!"
  done
  wait $P0; wait $P1
  tail -n 3 "$WORK"/captions_gpu*.log
  (cd "$CAP_ROOT" && python scripts/blip_caption_vqa.py --merge "$WORK/captions/captions_model0.json" \
    "$WORK/captions/captions_model1.json" --out "$WORK/captions/captions_crossfit.json")
  rm -rf "$HF_HOME"
fi

echo "== 4. train + test: seeds 0 and 1 in parallel (GPU 0 / GPU 1), then seed 2 =="
train() { python train_vqa.py --config "$CFG" --captions $CAPS --seed $1 --device cuda:$2 > "$WORK/train_s$1.log" 2>&1; }
train 0 0 & T0=$!
train 1 1 & T1=$!
wait $T0; wait $T1
train 2 0
grep -h -E "^\[" "$WORK"/train_s*.log
python summarize_vqa.py --runs "$WORK/runs" --out "$WORK/summary.md"

rm -rf "$WORK/features"   # Finglish: ~13 GB, lazem nist → output kochik
echo "VQA_FULL_${MODE^^}_DONE"
