"""Generic (question-independent) captions from BLIP-Large as released (COCO-caption fine-tuned, no training of ours).

Control for the paper: same starting checkpoint as our question-dependent captioner, but no QD fine-tuning and no
question prompt. One caption per image (beam 3), copied to every question of that image, written in the format the VQA
runner reads ({question_id: caption}).

Finglish: marhale 1 (caption): har GPU nesf-e image ha (shard k), resumable (har chand batch save mishe).
marhale 2 (expand): shard ha → caption-e har soal; check mikone ke hame-ye soal ha caption daran.

    # 1. caption the images of one part, shard k of 2 (one process per GPU)
    python scripts/generic_captions.py caption --part trainval --questions <train q.json> <val q.json> \
        --images-dir train2014=<dir> val2014=<dir> --shard 0 --num-shards 2 --device cuda:0 --out-dir <out>
    # 2. merge the shards and write captions_crossfit_generic.json / captions_test2015_generic.json
    python scripts/generic_captions.py expand --part trainval --questions <train q.json> <val q.json> --out-dir <out>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

TEMPLATE = {"train2014": "COCO_train2014_{:012d}.jpg", "val2014": "COCO_val2014_{:012d}.jpg",
            "test2015": "COCO_test2015_{:012d}.jpg"}
OUT_NAME = {"trainval": "captions_crossfit_generic.json", "test2015": "captions_test2015_generic.json"}


def load_questions(paths: List[str]) -> List[Tuple[str, int, int]]:
    """(image split, image_id, question_id) for every question; the split comes from the file's data_subtype."""
    out = []
    for p in paths:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        split = d.get("data_subtype") or next(s for s in TEMPLATE if s in Path(p).name)
        out += [(split, int(q["image_id"]), int(q["question_id"])) for q in d["questions"]]
    return out


def images_of(questions) -> List[Tuple[str, int]]:
    return sorted({(s, i) for s, i, _ in questions})


def shard_file(out_dir: Path, part: str, k: int) -> Path:
    return out_dir / f"generic_{part}_images_shard{k}.json"


def caption(args) -> None:
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, BlipForConditionalGeneration, BlipImageProcessor, BlipProcessor

    dirs = dict(x.split("=", 1) for x in args.images_dir)
    imgs = images_of(load_questions(args.questions))[args.shard::args.num_shards]
    out = shard_file(Path(args.out_dir), args.part, args.shard)
    done: Dict[str, str] = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    todo = [(s, i) for s, i in imgs if f"{s}/{i}" not in done]
    print(f"shard {args.shard}/{args.num_shards}: {len(imgs)} images, {len(done)} done, {len(todo)} to caption", flush=True)
    if not todo:
        return
    try:
        proc = BlipProcessor.from_pretrained(args.model)
        image_proc, tok = proc.image_processor, proc.tokenizer
    except (OSError, ValueError):  # tiny local test model: tokenizer only
        image_proc, tok = BlipImageProcessor(), AutoTokenizer.from_pretrained(args.model)
    device = torch.device(args.device)
    model = BlipForConditionalGeneration.from_pretrained(args.model).to(device).eval()
    if device.type == "cuda":
        model.half()

    class Images(Dataset):
        def __len__(self):
            return len(todo)

        def __getitem__(self, n):
            s, i = todo[n]
            img = Image.open(Path(dirs[s]) / TEMPLATE[s].format(i)).convert("RGB")
            return f"{s}/{i}", image_proc(images=img, return_tensors="pt")["pixel_values"][0]

    dl = DataLoader(Images(), batch_size=args.batch_size, num_workers=args.num_workers,
                    collate_fn=lambda b: ([x[0] for x in b], torch.stack([x[1] for x in b])))
    # Finglish: BLIP.generate token-e aval ro ba [DEC] avaz mikone va akhari ([SEP]) ro hazf → ba special token ha
    prompt = tok(args.prompt, return_tensors="pt") if args.prompt else None
    t0, n = time.time(), 0
    for b, (keys, pix) in enumerate(dl):
        pix = pix.to(device, dtype=next(model.parameters()).dtype)
        kw = dict(num_beams=args.beam, max_new_tokens=args.max_new_tokens)
        if prompt is not None:  # Finglish: "a picture of" → BLIP-large dige "arafed …" nemineveese
            kw["input_ids"] = prompt["input_ids"].repeat(len(keys), 1).to(device)
        with torch.no_grad():
            ids = model.generate(pixel_values=pix, **kw)
        for k, text in zip(keys, tok.batch_decode(ids, skip_special_tokens=True)):
            text = text.strip()
            if text.startswith("[DEC]"):
                text = text[len("[DEC]"):].strip()
            if args.prompt and text.lower().startswith(args.prompt.lower()):
                text = text[len(args.prompt):].strip()
            done[k] = text
        n += len(keys)
        if (b + 1) % args.save_every == 0 or n == len(todo):
            out.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
            rate = n / max(1e-6, time.time() - t0)
            print(f"  {len(done)}/{len(imgs)} images, {rate:.1f} img/s, eta {(len(todo) - n) / max(rate, 1e-6) / 60:.0f} min",
                  flush=True)
    out.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")


def expand(args) -> None:
    qs = load_questions(args.questions)
    out_dir = Path(args.out_dir)
    caps: Dict[str, str] = {}
    for f in sorted(out_dir.glob(f"generic_{args.part}_images_shard*.json")):
        caps.update(json.loads(f.read_text(encoding="utf-8")))
    imgs = images_of(qs)
    miss = [f"{s}/{i}" for s, i in imgs if f"{s}/{i}" not in caps]
    empty = sum(1 for s, i in imgs if not caps.get(f"{s}/{i}", "x").strip())
    res = {str(q): caps.get(f"{s}/{i}", "") for s, i, q in qs}
    path = out_dir / OUT_NAME[args.part]
    path.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    words = [len(c.split()) for c in caps.values()]
    print(f"{args.part}: {len(qs)} questions on {len(imgs)} images; images missing {len(miss)}, empty captions {empty}; "
          f"mean {sum(words) / max(1, len(words)):.1f} words -> {path}")
    for s, i in imgs[:5]:
        print(f"  {s}/{i}: {caps.get(f'{s}/{i}')}")
    if miss:
        raise SystemExit(f"{len(miss)} images have no caption (e.g. {miss[:3]}): run the caption step again")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("step", choices=["caption", "expand"])
    p.add_argument("--part", choices=["trainval", "test2015"], required=True)
    p.add_argument("--questions", nargs="+", required=True, help="VQA v2 question JSON files of this part")
    p.add_argument("--images-dir", nargs="*", default=[], help="split=dir, e.g. train2014=/path val2014=/path")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model", default="Salesforce/blip-image-captioning-large")
    p.add_argument("--prompt", default="a picture of", help='text prefix for generation ("" = none); stripped from the output')
    p.add_argument("--beam", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--save-every", type=int, default=100, help="batches between saves")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    caption(args) if args.step == "caption" else expand(args)


if __name__ == "__main__":
    main()
