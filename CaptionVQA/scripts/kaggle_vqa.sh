#!/usr/bin/env bash
# VQA on Kaggle (2× T4), everything in one notebook: split → BUTD features → (BLIP captions) → train 3 seeds → test.
# Run from the checkout:  cd <repo>/CaptionVQA && MODE=baseline bash scripts/kaggle_vqa.sh
#   MODE=baseline : Bottom-Up Top-Down VQA without captions, seeds 0 1 2                 (~45 min)
#   MODE=blip     : the same model + BLIP captions, seeds 0 1 2, + oracle captions seed 0 (~75 min)
# The two modes are independent → run them at the same time in two notebooks.
#
# Inputs (Add Input):
#   - mnhlfuch/vqa-v2                 (VQA v2 questions/annotations, COCO train2014 + val2014 images)
#   - questiondependentcaptions       (the QD caption JSON: which images the captioner was trained on)
#   - MODE=blip only: the BLIP training notebook's output (Add Input → Notebooks) with model_b/b_blip_full/best.pt
# Internet on (BUTD weights; MODE=blip also BLIP base weights).
# Output: /kaggle/working/vqa_<MODE>/runs/<nocap|blip|oracle>_s<seed>/{results.json,history.json,test_predictions.json}
#         and /kaggle/working/vqa_<MODE>/summary.md  → bring back the whole vqa_<MODE> folder.
set -euo pipefail
MODE=${MODE:-baseline}
[[ "$MODE" == "baseline" || "$MODE" == "blip" ]] || { echo "MODE must be baseline or blip"; exit 1; }
W=/kaggle/working
WORK=$W/vqa_$MODE
CFG=$W/configs/vqa_$MODE.yaml
CAP_ROOT=../QuestionDependentImageCaptioner
mkdir -p "$WORK" "$W/configs"
export PYTHONIOENCODING=utf-8

find1() { find -L /kaggle/input -name "$1" -type f 2>/dev/null | head -1; }
TRQ=$(find1 v2_OpenEnded_mscoco_train2014_questions.json)
TRA=$(find1 v2_mscoco_train2014_annotations.json)
VAQ=$(find1 v2_OpenEnded_mscoco_val2014_questions.json)
VAA=$(find1 v2_mscoco_val2014_annotations.json)
TRIMG=$(dirname "$(find1 COCO_train2014_000000000009.jpg)")
VAIMG=$(dirname "$(find1 COCO_val2014_000000000042.jpg)")
QD=$(cd "$CAP_ROOT" && python scripts/kaggle_find_qd.py configs/b_blip_full.yaml)
for v in TRQ TRA VAQ VAA; do [ -n "${!v}" ] || { echo "VQA v2 file missing ($v) under /kaggle/input"; exit 1; }; done
[ "$TRIMG" != "." ] && [ "$VAIMG" != "." ] || { echo "COCO train2014/val2014 images not found"; exit 1; }
printf 'MODE %s\nQD %s\nquestions %s | %s\nimages %s | %s\n' "$MODE" "$QD" "$TRQ" "$VAQ" "$TRIMG" "$VAIMG"

python - "$CFG" "$WORK" "$QD" "$TRQ" "$TRA" "$VAQ" "$VAA" "$TRIMG" "$VAIMG" <<'EOF'
import sys, yaml
dst, work, qd, trq, tra, vaq, vaa, trimg, vaimg = sys.argv[1:]
cfg = yaml.safe_load(open("configs/blip.yaml"))
cfg.update(work_dir=work, qd_captions_json=qd, train_questions_json=trq, train_annotations_json=tra,
           val_questions_json=vaq, val_annotations_json=vaa, train_images_dir=trimg, val_images_dir=vaimg,
           caption_requests_json=f"{work}/caption_requests.json")
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
print("config", dst)
EOF

echo "== 1. split =="
python make_split.py --config "$CFG"
# Finglish: split bayad dagighan hamoon-e repo bashe (data/blip_caption_requests.json, hamoon ke caption ha baraye oon sakhte mishan)
python - "$WORK/caption_requests.json" <<'EOF'
import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open("data/blip_caption_requests.json"))
assert [r["question_id"] for r in a] == [r["question_id"] for r in b], "split differs from the committed one"
print(f"split identical to the committed split ({len(a)} questions)")
EOF

echo "== 2. BUTD weights + region features (train on GPU 0, dev+test on GPU 1) =="
BUTD_URL=https://s3.amazonaws.com/models.huggingface.co/bert/unc-nlp/frcnn-vg-finetuned
BUTD_SHA=d3bf8ad74828a244a95d85e17a842cc0ac75419a53e4e81ee68386f192bb24c5
WD=$CAP_ROOT/outputs/butd_weights
if [ ! -f "$WD/pytorch_model.bin" ]; then
  mkdir -p "$WD"
  curl -sfL --retry 3 -o "$WD/config.yaml" "$BUTD_URL/config.yaml"
  curl -sfL --retry 3 -o "$WD/pytorch_model.bin" "$BUTD_URL/pytorch_model.bin"
  echo "$BUTD_SHA  $WD/pytorch_model.bin" | sha256sum -c --quiet && echo "BUTD weights OK (sha256)"
fi
python prepare_regions.py --config "$CFG" --splits train --device cuda:0 > "$WORK/regions_train.log" 2>&1 &
R0=$!
python prepare_regions.py --config "$CFG" --splits dev,test --device cuda:1 > "$WORK/regions_devtest.log" 2>&1 &
R1=$!
wait $R0; wait $R1
grep -h "images →\|images ->" "$WORK"/regions_*.log || tail -n 3 "$WORK"/regions_*.log

if [ "$MODE" = "blip" ]; then
  echo "== 3. BLIP captions =="
  (cd "$CAP_ROOT" && bash scripts/kaggle_captions_vqa.sh)
  mkdir -p "$WORK/captions"
  cp "$W/captions_vqa/captions_blip.json" "$WORK/captions/captions_blip.json"
fi

echo "== 4. train + test (3 seeds) =="
CAPS=none; [ "$MODE" = "blip" ] && CAPS=blip
for seed in 0 1 2; do
  python train_vqa.py --config "$CFG" --captions $CAPS --seed $seed 2>&1 | grep -E "^\[|^ep"
done
if [ "$MODE" = "blip" ]; then
  python train_vqa.py --config "$CFG" --captions oracle --seed 0 2>&1 | grep -E "^\[|^ep"
fi
python summarize_vqa.py --config "$CFG"

# Finglish: feature ha (~1.5 GB) lazem nistan → output kochik
rm -rf "$WORK/features" "$W/captions_vqa/captions_shard"*.json
echo "VQA_${MODE^^}_DONE"
