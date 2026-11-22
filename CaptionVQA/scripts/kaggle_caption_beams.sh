#!/usr/bin/env bash
# Captions with beam scores from the two cross-fit fold captioners (beam 3, all 3 beams kept):
#   {question_id: {"c": [3 captions], "s": [beam-search score], "mean": [mean token log-prob], "min": [min token log-prob]}}
# plus a plain {question_id: beam-1 caption} file for the VQA runner.
# Run from the checkout:  cd <repo>/CaptionVQA && CAPTIONER=large PART=trainval bash scripts/kaggle_caption_beams.sh
#
#   CAPTIONER=base   BLIP-base fold models   (model_b/b_blip_fold0/best.pt,       model_b/b_blip_fold1/best.pt)
#   CAPTIONER=large  BLIP-large fold models  (model_b/b_blip_fold0_large/best.pt, model_b/b_blip_fold1_large/best.pt)
#   PART=trainval    all 658,111 train2014 + val2014 questions, cross-fitted (a fold model never captions its own images)
#   PART=test2015    all 447,793 test2015 questions (test-dev + test-standard); neither fold model saw these images,
#                    each image goes to fold model (image_id mod 2), as val2014 is split half/half
# Inputs: mnhlfuch/vqa-v2 (images + questions; for test2015 the test2015 images and
#   v2_OpenEnded_mscoco_test2015_questions.json), questiondependentcaptions, the two fold captioner outputs.
#   Optional (trainval): a caption_requests.json (658k) to skip the split, a captions_crossfit.json for the beam-1 check.
# GPU T4 x2, internet on. Resumable: attach this notebook's own earlier output.
# Output: /kaggle/working/caption_beams_<CAPTIONER>/
#   trainval: captions_beams.json, captions_crossfit_<CAPTIONER>.json      test2015: captions_beams_test2015.json,
#   captions_test2015_<CAPTIONER>.json
set -euo pipefail
CAPTIONER=${CAPTIONER:-base}
PART=${PART:-trainval}
[[ "$CAPTIONER" == "base" || "$CAPTIONER" == "large" ]] || { echo "CAPTIONER must be base or large"; exit 1; }
[[ "$PART" == "trainval" || "$PART" == "test2015" ]] || { echo "PART must be trainval or test2015"; exit 1; }
SUF=$([ "$CAPTIONER" = large ] && echo "_large" || echo "")
BASE_CFG=$([ "$CAPTIONER" = large ] && echo configs/b_blip_fold_large.yaml || echo configs/b_blip_full.yaml)
W=/kaggle/working
OUT=$W/caption_beams_$CAPTIONER
CAP_ROOT=../QuestionDependentImageCaptioner
mkdir -p "$OUT" "$W/configs"
export PYTHONIOENCODING=utf-8
export HF_HOME=$W/hf_cache
START=$SECONDS
TAG=$([ "$PART" = trainval ] && echo "" || echo "_test2015")

read -r NGPU <<<"$(python -c "import torch; print(torch.cuda.device_count())")"
[ "$NGPU" -ge 2 ] || { echo "Need 2 GPUs (found $NGPU): set Settings → Accelerator → GPU T4 x2"; exit 1; }
find1() { find -L /kaggle/input -name "$1" -type f 2>/dev/null | head -1; }
# Finglish: checkpoint ha: hame best.pt ha, folder-e dagigh-e b_blip_fold<k><SUF> (na _v2…).
# Ya mostaghim: FOLD0_CKPT=/kaggle/input/... FOLD1_CKPT=/kaggle/input/...
mapfile -t BEST < <(find -L /kaggle/input -name 'best.pt' 2>/dev/null; find /kaggle/input -name 'best.pt' 2>/dev/null)
C0=${FOLD0_CKPT:-$(printf '%s\n' "${BEST[@]}" | grep -E "/b_blip_fold0${SUF}/best\.pt$" | head -1 || true)}
C1=${FOLD1_CKPT:-$(printf '%s\n' "${BEST[@]}" | grep -E "/b_blip_fold1${SUF}/best\.pt$" | head -1 || true)}
if [ -z "$C0" ] || [ -z "$C1" ] || [ ! -f "$C0" ] || [ ! -f "$C1" ]; then
  echo "fold captioners not found (need .../b_blip_fold0${SUF}/best.pt and .../b_blip_fold1${SUF}/best.pt)."
  echo "best.pt files under /kaggle/input:"; printf '  %s\n' "${BEST[@]:-none}"
  echo "input folders (depth 4):"; find /kaggle/input -maxdepth 4 -type d 2>/dev/null | head -40
  echo "set the paths by hand: FOLD0_CKPT=<path> FOLD1_CKPT=<path> bash scripts/kaggle_caption_beams.sh"
  exit 1
fi
echo "captioner $CAPTIONER | part $PART | fold models: $C0 | $C1"
TRIMG=$(dirname "$(find1 COCO_train2014_000000000009.jpg)")
[ "$TRIMG" != "." ] || { echo "COCO train2014 images not found"; exit 1; }
IMGARGS=()

echo "== 1. caption requests =="
if [ "$PART" = trainval ]; then
  VAIMG=$(dirname "$(find1 COCO_val2014_000000000042.jpg)")
  [ "$VAIMG" != "." ] || { echo "COCO val2014 images not found"; exit 1; }
  IMGARGS=(--val-images-dir "$VAIMG")
  REQ=""
  while IFS= read -r f; do
    n=$(python -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$f" 2>/dev/null || echo 0)
    [ "$n" = "658111" ] && { REQ=$f; break; }
  done < <(find -L /kaggle/input -name 'caption_requests.json' -type f 2>/dev/null)
  if [ -z "$REQ" ]; then
    # Finglish: mesl-e kaggle_vqa_strong.sh: split-e train2014 + val2014 → hamoon caption_requests
    TRQ=$(find1 v2_OpenEnded_mscoco_train2014_questions.json); TRA=$(find1 v2_mscoco_train2014_annotations.json)
    VAQ=$(find1 v2_OpenEnded_mscoco_val2014_questions.json);   VAA=$(find1 v2_mscoco_val2014_annotations.json)
    QD=$(cd "$CAP_ROOT" && python scripts/kaggle_find_qd.py configs/b_blip_full.yaml)
    python - "$W/configs/beams_split.yaml" "$OUT/work" "$QD" "$TRQ" "$TRA" "$VAQ" "$VAA" "$TRIMG" "$VAIMG" <<'EOF'
import sys, yaml
dst, work, qd, trq, tra, vaq, vaa, trimg, vaimg = sys.argv[1:]
cfg = yaml.safe_load(open("configs/blip_strong_b64.yaml"))
cfg.update(work_dir=work, qd_captions_json=qd, train_questions_json=trq, train_annotations_json=tra,
           val_questions_json=vaq, val_annotations_json=vaa, train_images_dir=trimg, val_images_dir=vaimg,
           caption_requests_json=f"{work}/caption_requests.json")
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
EOF
    python make_split.py --config "$W/configs/beams_split.yaml"
    REQ=$OUT/work/caption_requests.json
  fi
  python - "$REQ" <<'EOF'
import json, sys
req = json.load(open(sys.argv[1])); small = json.load(open("data/blip_caption_requests.json"))
for s in ("dev", "test"):
    assert [r["question_id"] for r in req if r["vqa_split"] == s] == [r["question_id"] for r in small if r["vqa_split"] == s], s
print(f"requests {sys.argv[1]}: {len(req)} questions, dev/test identical to all VQA experiments "
      f"(model 0: {sum(r['caption_model'] == 0 for r in req)}, model 1: {sum(r['caption_model'] == 1 for r in req)})")
EOF
else
  TEQ=$(find1 v2_OpenEnded_mscoco_test2015_questions.json)
  TEIMG=$(dirname "$(find1 COCO_test2015_000000000001.jpg)")
  [ -n "$TEQ" ] || { echo "v2_OpenEnded_mscoco_test2015_questions.json not found: attach a dataset with the VQA v2 test2015 questions"; exit 1; }
  [ "$TEIMG" != "." ] || { echo "COCO test2015 images not found: attach a dataset with the test2015 images"; exit 1; }
  IMGARGS=(--test-images-dir "$TEIMG")
  REQ=$OUT/test2015_requests.json
  python - "$TEQ" "$REQ" <<'EOF'
import json, sys
q = json.load(open(sys.argv[1]))["questions"]
# Finglish: test2015: hich captioner in image ha ro nadide → image_id mod 2 (nesf/nesf, mesl-e val2014)
req = [{"question_id": x["question_id"], "image_id": x["image_id"], "image_split": "test", "question": x["question"],
        "vqa_split": "test2015", "caption_model": x["image_id"] % 2} for x in q]
json.dump(req, open(sys.argv[2], "w"))
print(f"test2015: {len(req)} questions on {len({r['image_id'] for r in req})} images "
      f"(model 0: {sum(r['caption_model'] == 0 for r in req)}, model 1: {sum(r['caption_model'] == 1 for r in req)})")
EOF
fi

echo "== 2. beam captions + scores, fold model k on GPU k =="
python - "$TRIMG" "$W/configs/b_captions_$CAPTIONER.yaml" "$CAP_ROOT/$BASE_CFG" <<'EOF'
import sys, yaml
img, dst, base = sys.argv[1:]
cfg = yaml.safe_load(open(base))
cfg.update(train_images_dir=img, ddp=False, num_workers=2, eval_batch_size=96)
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
print(f"captioning config: {cfg['blip_model']}")
EOF
for k in 0 1; do
  prev=$(find -L /kaggle/input -path "*caption_beams_$CAPTIONER/captions_beams${TAG}_model$k.json" 2>/dev/null | head -1 || true)
  [ -n "$prev" ] && [ ! -f "$OUT/captions_beams${TAG}_model$k.json" ] && cp "$prev" "$OUT/" && echo "resume captions_beams${TAG}_model$k.json"
done
for k in 0 1; do
  CK=$([ $k = 0 ] && echo "$C0" || echo "$C1")
  (cd "$CAP_ROOT" && python scripts/blip_caption_vqa.py --config "$W/configs/b_captions_$CAPTIONER.yaml" --ckpt "$CK" \
    --requests "$REQ" "${IMGARGS[@]}" --caption-model $k --device cuda:$k --beam 3 --with-scores \
    --out "$OUT/captions_beams${TAG}_model$k.json") > "$OUT/beams${TAG}_gpu$k.log" 2>&1 &
  eval "P$k=\$!"
done
wait $P0; wait $P1
grep -h -E "requests|captions in|  [0-9]+/[0-9]+ captions" "$OUT"/beams${TAG}_gpu*.log | tail -n 6

echo "== 3. merge + checks =="
(cd "$CAP_ROOT" && python scripts/blip_caption_vqa.py --merge "$OUT/captions_beams${TAG}_model0.json" \
  "$OUT/captions_beams${TAG}_model1.json" --out "$OUT/captions_beams${TAG}.json")
PLAIN=$([ "$PART" = trainval ] && echo "captions_crossfit_$CAPTIONER.json" || echo "captions_test2015_$CAPTIONER.json")
OLD=""
[ "$PART" = trainval ] && [ "$CAPTIONER" = base ] && OLD=$(find -L /kaggle/input -name 'captions_crossfit.json' -type f -printf '%s %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || true)
python - "$REQ" "$OUT/captions_beams${TAG}.json" "$OUT/$PLAIN" "${OLD:-}" <<'EOF'
import json, statistics, sys
req, new = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
miss = sum(str(r["question_id"]) not in new for r in req)
print(f"beam captions: {len(new)} for {len(req)} questions, missing {miss}")
s = [v["s"][0] for v in new.values()]
print(f"beam-1 score: mean {statistics.mean(s):.3f}, median {statistics.median(s):.3f}")
# Finglish: file-e sade {qid: caption-e beam 1} baraye kaggle_vqa_strong.sh (CAPTIONS_FILE=...)
json.dump({k: v["c"][0] for k, v in new.items()}, open(sys.argv[3], "w"), ensure_ascii=False)
print(f"beam-1 captions -> {sys.argv[3]}")
if sys.argv[4]:
    old = json.load(open(sys.argv[4]))
    keys = [k for k in new if k in old]
    same = sum(new[k]["c"][0] == old[k] for k in keys)
    print(f"beam 1 == caption already used by the VQA models: {same}/{len(keys)} ({100 * same / max(1, len(keys)):.2f}%)")
sys.exit(1 if miss else 0)
EOF
rm -f "$OUT/captions_beams${TAG}_model0.json" "$OUT/captions_beams${TAG}_model1.json"
rm -rf "$OUT/work" "$HF_HOME"
ls -la "$OUT"
echo "total $(( (SECONDS - START) / 60 )) min"
echo "CAPTION_BEAMS_DONE ($CAPTIONER, $PART)"
