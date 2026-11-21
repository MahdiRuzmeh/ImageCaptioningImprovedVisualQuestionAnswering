"""Zero-shot probe: how much does a pretrained BLIP-2 (no fine-tuning) know about our questions?

Finglish: hamoon 10,000 row-e QD-e fold 1 (seed 0) ke captioner haye dige roosh moghayese shodan
(eval_fold_captioner.py: current 58.10 AnswerHit, 75.10 polarity). BLIP-2 *pretrained* (faghat caption-e tasvir,
bedoon VQA) ba prompt-e soal javab mide; AnswerHit / YesNoPolarity / CIDEr ba hamoon score().
  - --prompt short:     "Question: {q} Short answer:"            (prompt-e VQA-e maghale-ye BLIP-2)
  - --prompt sentence:  "Question: {q} Answer in a full sentence:" (shabih-e caption haye QD)
Har GPU yek shard (--shard i/n); ba'd --merge shard ha ro yeki va score mikone.

    python scripts/blip2_probe.py --model Salesforce/blip2-opt-2.7b --prompt short --qd <QD> \
        --folds ../CaptionVQA/data/crossfit_folds.json --images-dir <train2014> --shard 0/2 --device cuda:0 --out s0.json
    python scripts/blip2_probe.py --merge s0.json s1.json --qd <QD> --folds ... --name opt_short --out opt_short.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eval_fold_captioner import report, sample_rows  # noqa: E402

PROMPTS = {"short": "Question: {q} Short answer:", "sentence": "Question: {q} Answer in a full sentence:"}


def generate(args, rows):
    import torch
    from PIL import Image
    from transformers import Blip2ForConditionalGeneration, Blip2Processor

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    proc = Blip2Processor.from_pretrained(args.model)
    proc.tokenizer.padding_side = "left"  # Finglish: OPT (decoder-only) → padding chap baraye generate-e batch
    model = Blip2ForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    model.eval()
    i, n = (int(x) for x in args.shard.split("/"))
    part = rows[i::n]
    out_path = Path(args.out)
    out = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    todo = [r for r in part if str(r["question_id"]) not in out]
    print(f"{args.model} | prompt {args.prompt} | shard {i}/{n}: {len(part)} rows, {len(todo)} to do", flush=True)
    t0, bs = time.time(), args.batch_size
    for k in range(0, len(todo), bs):
        b = todo[k : k + bs]
        imgs = [Image.open(Path(args.images_dir) / f"COCO_train2014_{int(r['image_id']):012d}.jpg").convert("RGB")
                for r in b]
        enc = proc(images=imgs, text=[PROMPTS[args.prompt].format(q=r["question"]) for r in b],
                   return_tensors="pt", padding=True)
        enc = {kk: (v.to(device, dtype) if v.is_floating_point() else v.to(device)) for kk, v in enc.items()}
        with torch.no_grad():
            seq = model.generate(**enc, max_new_tokens=args.max_new_tokens, num_beams=args.beam, do_sample=False)
        # Finglish: OPT: khoroji = prompt + javab → faghat token haye jadid; T5 (encoder-decoder): faghat javab
        if not model.config.use_decoder_only_language_model:
            new = seq
        else:
            new = seq[:, enc["input_ids"].shape[1]:]
        for r, text in zip(b, proc.tokenizer.batch_decode(new, skip_special_tokens=True)):
            out[str(r["question_id"])] = text.strip()
        if (k // bs) % 20 == 0 or k + bs >= len(todo):
            tmp = out_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
            tmp.replace(out_path)
            rate = (k + len(b)) / max(time.time() - t0, 1e-9)
            print(f"  {k + len(b)}/{len(todo)} ({rate:.1f}/s, ~{(len(todo) - k - len(b)) / max(rate, 1e-9) / 60:.0f} min left)",
                  flush=True)
    for r in part[:3]:
        print(f"  Q: {r['question']}  ->  {out[str(r['question_id'])]}   (answer: {r['answer']})", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="Salesforce/blip2-opt-2.7b")
    p.add_argument("--prompt", choices=list(PROMPTS), default="short")
    p.add_argument("--qd", required=True)
    p.add_argument("--folds", required=True)
    p.add_argument("--fold", type=int, default=0, help="rows of fold 1 - fold (same rows as the fold-0 captioner eval)")
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--images-dir")
    p.add_argument("--shard", default="0/1")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--beam", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--merge", nargs="*", help="shard outputs → score")
    p.add_argument("--name", default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    rows = sample_rows(args.qd, args.folds, args.fold, args.n, args.seed)
    if not args.merge:
        generate(args, rows)
        return
    caps = {}
    for f in args.merge:
        caps.update(json.loads(Path(f).read_text(encoding="utf-8")))
    miss = [r for r in rows if str(r["question_id"]) not in caps]
    assert not miss, f"{len(miss)} rows have no output"
    preds = [caps[str(r["question_id"])] for r in rows]
    res = report(preds, rows, args.name or Path(args.out).stem)
    res.update(model=args.model, prompt=args.prompt, n=len(rows), seed=args.seed,
               predictions=[{"question_id": r["question_id"], "question": r["question"], "answer": r["answer"],
                             "gt": r["caption"], "pred": c} for r, c in zip(rows, preds)])
    Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
