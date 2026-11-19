#!/usr/bin/env bash
# Stage A + B on Kaggle (2× T4): MCAN-style VQA + GloVe, trained on train2014 + val2014 (643,110 questions).
# Same test as all earlier experiments (10,001 val2014 questions).
# Run from the checkout:  cd <repo>/CaptionVQA && MODE=baseline bash scripts/kaggle_vqa_strong.sh
#   MODE=baseline : stronger baseline, no captions
#   MODE=crossfit : the same model + cross-fitted BLIP captions
# Env: SEEDS="0 1 2"   seeds 0 and 1 run in parallel on the 2 GPUs; seed 2 only if time allows
#      CONFIG=configs/blip_strong.yaml   round 1b: configs/blip_strong_b64.yaml (batch 64)
#      CAPTION_DROPOUT=0.3               crossfit only: share of training questions whose caption is hidden
#      RUN=vqa_strong_<MODE>             output folder name, e.g. RUN=vqa_b64_crossfit_drop
#
# Inputs (Add Input):
#   - mnhlfuch/vqa-v2, questiondependentcaptions
#   - qd-butd-packed (train2014 features) and the val2014 cache from notebook 0 (butd_val2014/)
#   - a GloVe dataset with glove.*.300d.txt (glove.840B.300d.txt is preferred when both are attached)
#   - MODE=crossfit: a captions_crossfit.json from an earlier run (the largest one found is reused).
#     Only if some questions still have no caption: the fold notebooks' outputs (b_blip_fold0/1/best.pt).
# Internet on. Output: /kaggle/working/<RUN>/ (runs/, summary.md, captions/, logs).
set -euo pipefail
MODE=${MODE:-baseline}
SEEDS=${SEEDS:-"0 1 2"}
CONFIG=${CONFIG:-configs/blip_strong.yaml}
CAPTION_DROPOUT=${CAPTION_DROPOUT:-0}
RUN=${RUN:-vqa_strong_$MODE}
[[ "$MODE" == "baseline" || "$MODE" == "crossfit" ]] || { echo "MODE must be baseline or crossfit"; exit 1; }
W=/kaggle/working
WORK=$W/$RUN
CFG=$W/configs/$RUN.yaml
CAP_ROOT=../QuestionDependentImageCaptioner
mkdir -p "$WORK/captions" "$W/configs"
export PYTHONIOENCODING=utf-8
START=$SECONDS

read -r NGPU <<<"$(python -c "import torch; print(torch.cuda.device_count())")"
[ "$NGPU" -ge 2 ] || { echo "Need 2 GPUs (found $NGPU): set Settings → Accelerator → GPU T4 x2"; exit 1; }
find1() { find -L /kaggle/input -name "$1" -type f 2>/dev/null | head -1; }
TRQ=$(find1 v2_OpenEnded_mscoco_train2014_questions.json); TRA=$(find1 v2_mscoco_train2014_annotations.json)
VAQ=$(find1 v2_OpenEnded_mscoco_val2014_questions.json);   VAA=$(find1 v2_mscoco_val2014_annotations.json)
TRIMG=$(dirname "$(find1 COCO_train2014_000000000009.jpg)"); VAIMG=$(dirname "$(find1 COCO_val2014_000000000042.jpg)")
QD=$(cd "$CAP_ROOT" && python scripts/kaggle_find_qd.py configs/b_blip_full.yaml)
# Finglish: GloVe 840B (vocab-e bozorgtar) age attach shode, vagarna har glove*300d*.txt
GLOVE=$(find -L /kaggle/input -type f -iname 'glove.840B*300d*.txt' 2>/dev/null | head -1 || true)
[ -n "$GLOVE" ] || GLOVE=$(find -L /kaggle/input -type f -iname 'glove*300d*.txt' 2>/dev/null | head -1 || true)
CACHES=()
while IFS= read -r m; do CACHES+=("$(dirname "$m")"); done < <(find -L /kaggle/input -name meta.json -path '*butd_*' 2>/dev/null)
for v in TRQ TRA VAQ VAA; do [ -n "${!v}" ] || { echo "VQA v2 file missing ($v)"; exit 1; }; done
[ "$TRIMG" != "." ] && [ "$VAIMG" != "." ] || { echo "COCO train2014/val2014 images not found"; exit 1; }
[ -n "$GLOVE" ] || echo "WARNING: no glove*300d*.txt attached -> random word embeddings (weaker model)"
[ ${#CACHES[@]} -ge 2 ] || echo "WARNING: fewer than 2 BUTD caches attached (qd-butd-packed + qd-butd-val2014) -> extra extraction time"
# Finglish: feature ha (~18 GB) → /tmp age ja dare (/kaggle/working faghat 20 GB)
FREE_TMP=$(df -BG --output=avail /tmp | tail -1 | tr -dc '0-9')
FEAT=$([ "$FREE_TMP" -ge 30 ] && echo /tmp/vqa_strong_features || echo "$WORK/features")
echo "RUN $RUN | MODE $MODE | config $CONFIG | caption dropout $CAPTION_DROPOUT | seeds $SEEDS"
echo "QD $QD"
echo "GloVe ${GLOVE:-none}"
echo "feature caches: ${CACHES[*]:-none}"
echo "features -> $FEAT (/tmp free $FREE_TMP GB)"

python - "$CFG" "$WORK" "$QD" "$TRQ" "$TRA" "$VAQ" "$VAA" "$TRIMG" "$VAIMG" "$FEAT" "$GLOVE" "$CONFIG" "$CAPTION_DROPOUT" <<'EOF'
import sys, yaml
dst, work, qd, trq, tra, vaq, vaa, trimg, vaimg, feat, glove, base, drop = sys.argv[1:]
cfg = yaml.safe_load(open(base))
cfg.update(work_dir=work, qd_captions_json=qd, train_questions_json=trq, train_annotations_json=tra,
           val_questions_json=vaq, val_annotations_json=vaa, train_images_dir=trimg, val_images_dir=vaimg,
           features_dir=feat, glove_path=glove or None, caption_dropout=float(drop),
           caption_requests_json=f"{work}/caption_requests.json", captions_json=f"{work}/captions/captions_crossfit.json")
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
print(f"config {dst}: batch {cfg['batch_size']}, epochs {cfg['epochs']}, caption_dropout {cfg['caption_dropout']}")
EOF

echo "== 1. split (train2014 + val2014) =="
python make_split.py --config "$CFG"
python - "$WORK" <<'EOF'
import json, sys
req = json.load(open(f"{sys.argv[1]}/caption_requests.json")); small = json.load(open("data/blip_caption_requests.json"))
for s in ("dev", "test"):
    assert [r["question_id"] for r in req if r["vqa_split"] == s] == [r["question_id"] for r in small if r["vqa_split"] == s], s
print(f"dev/test identical to the earlier experiments; {len(req)} questions")
EOF

echo "== 2. BUTD features (copy from caches, extract the rest on 2 GPUs) =="
BUTD_URL=https://s3.amazonaws.com/models.huggingface.co/bert/unc-nlp/frcnn-vg-finetuned
BUTD_SHA=d3bf8ad74828a244a95d85e17a842cc0ac75419a53e4e81ee68386f192bb24c5
WD=$CAP_ROOT/outputs/butd_weights
if [ ! -f "$WD/pytorch_model.bin" ]; then
  mkdir -p "$WD"
  curl -sfL --retry 3 -o "$WD/config.yaml" "$BUTD_URL/config.yaml"
  curl -sfL --retry 3 -o "$WD/pytorch_model.bin" "$BUTD_URL/pytorch_model.bin"
  echo "$BUTD_SHA  $WD/pytorch_model.bin" | sha256sum -c --quiet && echo "BUTD weights OK (sha256)"
fi
PK=()
[ ${#CACHES[@]} -gt 0 ] && PK=(--packed-dir "${CACHES[@]}")
python prepare_regions.py --config "$CFG" --init-only "${PK[@]}" 2>&1 | grep -v "it/s"
for k in 0 1; do
  python prepare_regions.py --config "$CFG" --shard $k/2 --device cuda:$k > "$WORK/regions_gpu$k.log" 2>&1 &
  eval "R$k=\$!"
done
wait $R0; wait $R1
grep -h "shard" "$WORK"/regions_gpu*.log | grep -v "it/s" || true
echo "features ready after $(( (SECONDS - START) / 60 )) min"

CAPS=none
if [ "$MODE" = "crossfit" ]; then
  CAPS=blip
  echo "== 3. captions: reuse earlier captions, caption only what is missing =="
  # Finglish: bozorgtarin captions_crossfit.json dar input ha (e.g. 658k-e run-e ghabli)
  OLD=$(find -L /kaggle/input -name 'captions_crossfit.json' -type f -printf '%s %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || true)
  echo "earlier captions: ${OLD:-none (all 658k questions will be captioned, ~4 h)}"
  python - "$WORK" "${OLD:-}" <<'EOF'
import json, sys
work, old = sys.argv[1], sys.argv[2]
req = json.load(open(f"{work}/caption_requests.json"))
have = json.load(open(old)) if old else {}
todo = [r for r in req if str(r["question_id"]) not in have]
json.dump(todo, open(f"{work}/caption_requests_todo.json", "w"))
json.dump({str(r["question_id"]): have[str(r["question_id"])] for r in req if str(r["question_id"]) in have},
          open(f"{work}/captions/captions_reused.json", "w"))
print(f"reused {len(req) - len(todo)} captions, {len(todo)} to generate "
      f"(model 0: {sum(r['caption_model'] == 0 for r in todo)}, model 1: {sum(r['caption_model'] == 1 for r in todo)})")
open(f"{work}/n_todo.txt", "w").write(str(len(todo)))
EOF
  NTODO=$(cat "$WORK/n_todo.txt")
  if [ "$NTODO" = "0" ]; then
    # Finglish: hame caption ha az ghabl hastan → model haye fold lazem nistan
    cp "$WORK/captions/captions_reused.json" "$WORK/captions/captions_crossfit.json"
  else
    C0=$(find -L /kaggle/input -path '*model_b/b_blip_fold0/best.pt' 2>/dev/null | head -1 || true)
    C1=$(find -L /kaggle/input -path '*model_b/b_blip_fold1/best.pt' 2>/dev/null | head -1 || true)
    [ -n "$C0" ] && [ -n "$C1" ] || { echo "attach both fold notebook outputs (b_blip_fold0/best.pt, b_blip_fold1/best.pt)"; exit 1; }
    for k in 0 1; do
      prev=$(find -L /kaggle/input -path "*$RUN/captions/captions_model$k.json" 2>/dev/null | head -1 || true)
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
        --requests "$WORK/caption_requests_todo.json" --val-images-dir "$VAIMG" --caption-model $k --device cuda:$k \
        --out "$WORK/captions/captions_model$k.json") > "$WORK/captions_gpu$k.log" 2>&1 &
      eval "P$k=\$!"
    done
    wait $P0; wait $P1
    tail -n 2 "$WORK"/captions_gpu*.log
    (cd "$CAP_ROOT" && python scripts/blip_caption_vqa.py --merge "$WORK/captions/captions_reused.json" \
      "$WORK/captions/captions_model0.json" "$WORK/captions/captions_model1.json" --out "$WORK/captions/captions_crossfit.json")
    rm -rf "$HF_HOME"
  fi
  python - "$WORK" <<'EOF'
import json, sys
w = sys.argv[1]; req = json.load(open(f"{w}/caption_requests.json")); cap = json.load(open(f"{w}/captions/captions_crossfit.json"))
miss = sum(str(r["question_id"]) not in cap for r in req)
print(f"captions: {len(cap)} for {len(req)} questions, missing {miss}")
sys.exit(1 if miss else 0)
EOF
  rm -f "$WORK/captions/captions_reused.json" "$WORK/caption_requests_todo.json" "$WORK/n_todo.txt"
  echo "captions ready after $(( (SECONDS - START) / 60 )) min"
fi

echo "== 4. train + test =="
train() { python train_vqa.py --config "$CFG" --captions $CAPS --seed $1 --device cuda:$2 > "$WORK/train_s$1.log" 2>&1; }
read -r -a S <<<"$SEEDS"
train "${S[0]}" 0 & T0=$!
[ ${#S[@]} -ge 2 ] && { train "${S[1]}" 1 & T1=$!; }
wait $T0; [ ${#S[@]} -ge 2 ] && wait $T1
ELAPSED=$(( SECONDS - START ))
if [ ${#S[@]} -ge 3 ]; then
  # Finglish: seed-e sevom faghat age vaght (ta 11 saat) mimoone: zaman-e yek seed ~ (ELAPSED − amade-sazi)
  if [ $(( ELAPSED * 2 )) -lt $(( 11 * 3600 )) ]; then train "${S[2]}" 0; else echo "skip seed ${S[2]}: not enough time left (run it with SEEDS=\"${S[2]}\")"; fi
fi
grep -h -E "^\[" "$WORK"/train_s*.log
python summarize_vqa.py --runs "$WORK/runs" --out "$WORK/summary.md"
rm -rf "$FEAT" "$WORK/features"
echo "total $(( (SECONDS - START) / 60 )) min"
echo "VQA_STRONG_${MODE^^}_DONE ($RUN)"
