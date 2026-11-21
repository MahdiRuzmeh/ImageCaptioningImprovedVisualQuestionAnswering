#!/usr/bin/env bash
# Zero-shot probe of pretrained BLIP-2 (no fine-tuning, no VQA training) on the same 10,000 fold-1 QD rows used for
# every captioner comparison (current BLIP-base fold-0 captioner: AnswerHit 58.10, polarity 75.10).
# Run from the checkout:  cd <repo>/QuestionDependentImageCaptioner && bash scripts/kaggle_blip2_probe.sh
#   MODELS="Salesforce/blip2-opt-2.7b Salesforce/blip2-flan-t5-xl"   PROMPTS="short sentence"   (defaults)
# Inputs: mnhlfuch/vqa-v2 (train2014 images), questiondependentcaptions. GPU T4 x2, internet on (~16 GB per model).
# ~1.5–2.5 h. Output: /kaggle/working/blip2_probe/{<model>_<prompt>.json, summary.md}
set -euo pipefail
MODELS=${MODELS:-"Salesforce/blip2-opt-2.7b Salesforce/blip2-flan-t5-xl"}
PROMPTS=${PROMPTS:-"short sentence"}
W=/kaggle/working
OUT=$W/blip2_probe
mkdir -p "$OUT"
export PYTHONIOENCODING=utf-8
# Finglish: model ha bozorgan (~16 GB) → cache dar /tmp (na /kaggle/working ke 20 GB-e), ba'd az har model pak mishe
export HF_HOME=/tmp/hf_blip2
START=$SECONDS

read -r NGPU <<<"$(python -c "import torch; print(torch.cuda.device_count())")"
[ "$NGPU" -ge 2 ] || { echo "Need 2 GPUs (found $NGPU): set Settings → Accelerator → GPU T4 x2"; exit 1; }
QD=$(python scripts/kaggle_find_qd.py configs/b_blip_full.yaml)
IMG=$(dirname "$(find -L /kaggle/input -name COCO_train2014_000000000009.jpg -type f 2>/dev/null | head -1)")
[ "$IMG" != "." ] || { echo "COCO train2014 images not found"; exit 1; }
FOLDS=../CaptionVQA/data/crossfit_folds.json
echo "QD $QD | images $IMG | /tmp free $(df -BG --output=avail /tmp | tail -1 | tr -dc '0-9') GB"
python -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

for M in $MODELS; do
  TAG=$(basename "$M" | tr '.-' '__')
  echo "== $M: download =="
  python -c "from huggingface_hub import snapshot_download as s; s('$M', allow_patterns=['*.json','*.safetensors','*.model','*.txt'])" 2>&1 | tail -1
  for P in $PROMPTS; do
    NAME=${TAG}_$P
    [ -f "$OUT/$NAME.json" ] && { echo "$NAME already done"; continue; }
    for k in 0 1; do
      prev=$(find -L /kaggle/input -path "*blip2_probe/${NAME}_s$k.json" 2>/dev/null | head -1 || true)
      [ -n "$prev" ] && [ ! -f "$OUT/${NAME}_s$k.json" ] && cp "$prev" "$OUT/" && echo "resume ${NAME}_s$k.json"
      python scripts/blip2_probe.py --model "$M" --prompt "$P" --qd "$QD" --folds "$FOLDS" --images-dir "$IMG" \
        --shard $k/2 --device cuda:$k --out "$OUT/${NAME}_s$k.json" > "$OUT/${NAME}_gpu$k.log" 2>&1 &
      eval "G$k=\$!"
    done
    wait $G0; wait $G1
    tail -n 4 "$OUT/${NAME}_gpu0.log"
    python scripts/blip2_probe.py --merge "$OUT/${NAME}_s0.json" "$OUT/${NAME}_s1.json" --model "$M" --prompt "$P" \
      --qd "$QD" --folds "$FOLDS" --name "$NAME" --out "$OUT/$NAME.json" 2>&1 | grep -v -i warn
    rm -f "$OUT/${NAME}_s0.json" "$OUT/${NAME}_s1.json"
  done
  rm -rf "$HF_HOME"
  echo "$M done after $(( (SECONDS - START) / 60 )) min"
done

python - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = ["# BLIP-2 zero-shot probe (10,000 fold-1 QD rows, same rows as the captioner comparison)", "",
        "| Captioner | AnswerHit | number | other | YesNoPolarity | CIDEr |", "|---|---:|---:|---:|---:|---:|",
        "| BLIP-base fold-0 captioner (fine-tuned on QD; reference) | 58.10 | 44.97 | 62.77 | 75.10 | 6.39 |"]
for f in sorted(out.glob("*.json")):
    d = json.loads(f.read_text(encoding="utf-8")); o, t = d["overall"], d["per_answer_type"]
    rows.append(f"| {d['model']} zero-shot, prompt {d['prompt']} | {100 * o['AnswerHit']:.2f} | "
                f"{100 * t.get('number', {}).get('AnswerHit', 0):.2f} | {100 * t.get('other', {}).get('AnswerHit', 0):.2f} | "
                f"{100 * o['YesNoPolarity']:.2f} | {o['CIDEr']:.2f} |")
(out / "summary.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
print("\n".join(rows))
EOF
echo "total $(( (SECONDS - START) / 60 )) min"
echo "BLIP2_PROBE_DONE"
