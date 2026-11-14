"""Local smoke test only: tiny random-weight BLIP (no download) → outputs/tiny_blip.

Finglish: hamoon class-e BlipForConditionalGeneration + BertTokenizer ba vocab az kalamat-e QD JSON,
vali 2 layer / 64 dim. Faghat baraye test-e code-e train_blip.py (natije ma'na nadare).
    python scripts/make_tiny_blip.py --qd ../dataset/vqa_v2_qd_captions_full_train2014.json
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from transformers import BertTokenizer, BlipConfig, BlipForConditionalGeneration

p = argparse.ArgumentParser()
p.add_argument("--qd", required=True)
p.add_argument("--out", default="outputs/tiny_blip")
a = p.parse_args()
rows = json.loads(Path(a.qd).read_text(encoding="utf-8"))["annotations"][:5000]
words = Counter(w for r in rows for w in re.findall(r"[a-z0-9]+|[^\sa-z0-9]", (r["question"] + " " + r["caption"]).lower()))
special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[DEC]"]
vocab = special + ["question", "caption", ":"] + [w for w, _ in words.most_common() if w not in ("question", "caption", ":")]
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
(out / "vocab.txt").write_text("\n".join(dict.fromkeys(vocab)) + "\n", encoding="utf-8")
tk = BertTokenizer(str(out / "vocab.txt"), do_lower_case=True)
cfg = BlipConfig(
    text_config=dict(vocab_size=len(tk), hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=128, encoder_hidden_size=64, pad_token_id=0, sep_token_id=3,
                     bos_token_id=5, eos_token_id=3),
    vision_config=dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=2, intermediate_size=128,
                       image_size=384, patch_size=16),
)
BlipForConditionalGeneration(cfg).save_pretrained(out)
tk.save_pretrained(out)
print("tiny BLIP ->", out, "vocab", len(tk))
