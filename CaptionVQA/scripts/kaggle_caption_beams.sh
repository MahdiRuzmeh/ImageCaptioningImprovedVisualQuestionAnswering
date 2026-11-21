#!/usr/bin/env bash
# Round 3 (C1): captioner confidence. Re-caption all 658,111 VQA questions with the SAME two cross-fit fold captioners
# and the same beam search (beam 3), but keep all 3 beams and their scores:
#   {question_id: {"c": [3 captions], "s": [beam-search score], "mean": [mean token log-prob], "min": [min token log-prob]}}
# Beam 1 is the caption the VQA models already use (checked against captions_crossfit.json when it is attached).
# Run from the checkout:  cd <repo>/CaptionVQA && bash scripts/kaggle_caption_beams.sh
#
# Inputs (Add Input): mnhlfuch/vqa-v2 (train2014 + val2014 images, questions), questiondependentcaptions,
#   the two fold notebooks' outputs (model_b/b_blip_fold0/best.pt, model_b/b_blip_fold1/best.pt),
#   optional: a captions_crossfit.json (658k) for the beam-1 check, a caption_requests.json (658k) to skip the split.
# GPU T4 x2, internet on (BLIP base weights). ~4.5 h. Resumable: attach this notebook's own earlier output.
# Output: /kaggle/working/caption_beams/captions_beams.json (~200 MB) + logs.
set -euo pipefail
W=/kaggle/working
OUT=$W/caption_beams
CAP_ROOT=../QuestionDependentImageCaptioner
mkdir -p "$OUT" "$W/configs"
export PYTHONIOENCODING=utf-8
export HF_HOME=$W/hf_cache
START=$SECONDS

read -r NGPU <<<"$(python -c "import torch; print(torch.cuda.device_count())")"
[ "$NGPU" -ge 2 ] || { echo "Need 2 GPUs (found $NGPU): set Settings → Accelerator → GPU T4 x2"; exit 1; }
find1() { find -L /kaggle/input -name "$1" -type f 2>/dev/null | head -1; }
TRIMG=$(dirname "$(find1 COCO_train2014_000000000009.jpg)"); VAIMG=$(dirname "$(find1 COCO_val2014_000000000042.jpg)")
[ "$TRIMG" != "." ] && [ "$VAIMG" != "." ] || { echo "COCO train2014/val2014 images not found"; exit 1; }
C0=$(find -L /kaggle/input -path '*model_b/b_blip_fold0/best.pt' 2>/dev/null | head -1 || true)
C1=$(find -L /kaggle/input -path '*model_b/b_blip_fold1/best.pt' 2>/dev/null | head -1 || true)
[ -n "$C0" ] && [ -n "$C1" ] || { echo "attach both fold notebook outputs (b_blip_fold0/best.pt, b_blip_fold1/best.pt)"; exit 1; }
echo "fold models: $C0 | $C1"

echo "== 1. caption requests (658,111 questions, which fold model captions which question) =="
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

echo "== 2. beam captions + scores, fold model k on GPU k =="
python - "$TRIMG" "$W/configs/b_captions.yaml" <<'EOF'
import sys, yaml
img, dst = sys.argv[1:]
cfg = yaml.safe_load(open("../QuestionDependentImageCaptioner/configs/b_blip_full.yaml"))
cfg.update(train_images_dir=img, ddp=False, num_workers=2, eval_batch_size=96)
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
EOF
for k in 0 1; do
  prev=$(find -L /kaggle/input -path "*caption_beams/captions_beams_model$k.json" 2>/dev/null | head -1 || true)
  [ -n "$prev" ] && [ ! -f "$OUT/captions_beams_model$k.json" ] && cp "$prev" "$OUT/" && echo "resume captions_beams_model$k.json"
done
for k in 0 1; do
  CK=$([ $k = 0 ] && echo "$C0" || echo "$C1")
  (cd "$CAP_ROOT" && python scripts/blip_caption_vqa.py --config "$W/configs/b_captions.yaml" --ckpt "$CK" \
    --requests "$REQ" --val-images-dir "$VAIMG" --caption-model $k --device cuda:$k --beam 3 --with-scores \
    --out "$OUT/captions_beams_model$k.json") > "$OUT/beams_gpu$k.log" 2>&1 &
  eval "P$k=\$!"
done
wait $P0; wait $P1
grep -h -E "requests|captions in|  [0-9]+/[0-9]+ captions" "$OUT"/beams_gpu*.log | tail -n 6

echo "== 3. merge + checks =="
(cd "$CAP_ROOT" && python scripts/blip_caption_vqa.py --merge "$OUT/captions_beams_model0.json" \
  "$OUT/captions_beams_model1.json" --out "$OUT/captions_beams.json")
OLD=$(find -L /kaggle/input -name 'captions_crossfit.json' -type f -printf '%s %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || true)
python - "$REQ" "$OUT/captions_beams.json" "${OLD:-}" <<'EOF'
import json, statistics, sys
req, new = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
miss = sum(str(r["question_id"]) not in new for r in req)
print(f"beam captions: {len(new)} for {len(req)} questions, missing {miss}")
s = [v["s"][0] for v in new.values()]
print(f"beam-1 score: mean {statistics.mean(s):.3f}, median {statistics.median(s):.3f}")
if sys.argv[3]:
    old = json.load(open(sys.argv[3]))
    keys = [k for k in new if k in old]
    same = sum(new[k]["c"][0] == old[k] for k in keys)
    print(f"beam 1 == caption already used by the VQA models: {same}/{len(keys)} ({100 * same / max(1, len(keys)):.2f}%)")
sys.exit(1 if miss else 0)
EOF
rm -f "$OUT/captions_beams_model0.json" "$OUT/captions_beams_model1.json"
rm -rf "$OUT/work" "$HF_HOME"
ls -la "$OUT"
echo "total $(( (SECONDS - START) / 60 )) min"
echo "CAPTION_BEAMS_DONE"
