"""Upper-bound caption features: COCO human captions in place of generated ones.

Har (image, soal): 3 ta az 5 human caption-e COCO (bishtarin overlap-e kalame ba
soal, faghat soal — na javab — pas test-time ham ghabel-e estefade ast) → hamoon
format-e ``captions_{split}.pt`` (tokens / scores / hidden) ta A/B/C bedoon taghir
estefade konan. ``hidden`` va ``scores`` ba teacher forcing-e hamoon captioner-e
frozen (kaggle_mini) roye human caption sakhte mishan. Kalame haye kharej az
vocab-e captioner → ``<unk>``.

    python prepare_human_captions.py --config configs/small.yaml
    → work_dir/features/captions_{train,dev,test}_human.pt
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

from common import load_config, read_json, tok, write_json
from prepare_features import SPLITS, encode_question, load_frozen_captioner, pad_batch, strip_bos_eos

STOP = {"a", "an", "the", "is", "are", "was", "were", "be", "of", "in", "on", "at", "to", "for", "with",
        "and", "or", "this", "that", "these", "those", "it", "there", "what", "which", "who", "how",
        "many", "much", "does", "do", "did", "can", "could", "picture", "image", "photo", "shown", "any"}
N_CAPS = 3
LENGTH_ALPHA = 0.7  # hamoon length normalization-e beam_search_all


def load_coco_captions(cfg: dict) -> Dict[int, List[str]]:
    caps: Dict[int, List[str]] = defaultdict(list)
    for key in ("coco_train_captions_json", "coco_val_captions_json"):
        for a in read_json(cfg[key])["annotations"]:
            caps[a["image_id"]].append(a["caption"])
    return caps


def pick_captions(question: str, caps: List[str]) -> List[str]:
    """Top-3 caption ba overlap-e kalame haye mohtava-ye soal (tie → tartib-e COCO)."""
    q = {w for w in tok(question) if w not in STOP}
    ranked = sorted(range(len(caps)), key=lambda i: (-len(q & set(tok(caps[i]))), i))
    return [caps[i] for i in ranked[:N_CAPS]]


def encode_caption(text: str, stoi: Dict[str, int], max_len: int) -> List[int]:
    unk = stoi["<unk>"]
    return [1] + [stoi.get(w, unk) for w in tok(text)[: max_len - 2]] + [2]


@torch.no_grad()
def teacher_force(captioner, s: dict, seqs: torch.Tensor):
    """seqs (n, T) ba BOS/EOS → (length-normalized log-prob (n,), decoder states (n, T, d))."""
    logp, h = captioner._decode(seqs, s)
    tgt = seqs[:, 1:]
    lp = logp[:, :-1].float().gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * (tgt != 0)
    lengths = (seqs != 0).sum(1).clamp(min=1).float()
    return lp.sum(1) / lengths.pow(LENGTH_ALPHA), h


def caption_split(cfg: dict, name: str, coco: Dict[int, List[str]], captioner, vocab, q_vocab, device) -> None:
    work = Path(cfg["work_dir"])
    out = work / "features" / f"captions_{name}_human.pt"
    if out.exists():
        print(f"[human] {out} exists, skip")
        return
    recs = read_json(work / "split" / f"{name}.json")
    cache_dir = str(work / "region_cache" / ("train" if name == "train" else "val"))
    max_len = int(cfg["caption_max_len"])
    L = max_len - 1
    all_tok, all_score, all_hid, texts = [], [], [], []
    bs = 128
    for i in tqdm(range(0, len(recs), bs), desc=f"human/{name}"):
        chunk = recs[i : i + bs]
        picked = [pick_captions(r["question"], coco[r["image_id"]]) for r in chunk]
        texts.extend(picked)
        q = pad_batch([encode_question(r["question"], q_vocab, int(cfg["max_question_len"])) for r in chunk]).to(device)
        iids = torch.tensor([r["image_id"] for r in chunk], device=device)
        dummy = torch.zeros((len(chunk), 3, 1, 1), device=device)
        s = captioner._encode(dummy, q, iids, cache_dir, False)
        beams, scores = [], []
        for k in range(N_CAPS):
            seqs = pad_batch([encode_caption(p[k], vocab.stoi, max_len) for p in picked]).to(device)
            sc, h = teacher_force(captioner, s, seqs)
            beams.append(seqs)
            scores.append(sc)
            if k == 0:  # Finglish: mesl-e generated: decoder state faghat baraye caption-e aval
                hid = h[:, 1:]
                hid = torch.nn.functional.pad(hid, (0, 0, 0, max(0, L - hid.size(1))))[:, :L]
        T = max(b.size(1) for b in beams)
        seqs = torch.stack([torch.nn.functional.pad(b, (0, T - b.size(1))) for b in beams], 1)
        tokens = strip_bos_eos(seqs, L)                       # (n, 3, L)
        hid = hid * (tokens[:, 0] != 0).unsqueeze(-1)
        all_tok.append(tokens.short().cpu())
        all_score.append(torch.stack(scores, 1).float().cpu())
        all_hid.append(hid.half().cpu())
    tokens = torch.cat(all_tok)
    torch.save({
        "question_ids": [r["question_id"] for r in recs],
        "tokens": tokens,
        "scores": torch.cat(all_score),
        "hidden": torch.cat(all_hid),
    }, out)
    sample = []
    for k in range(0, len(recs), max(1, len(recs) // 30)):
        caps = [" ".join(vocab.itos[t] for t in row.tolist() if t > 0) for row in tokens[k]]
        sample.append({"question": recs[k]["question"], "answers": recs[k]["answers"][:3],
                       "coco": texts[k], "captions": caps})
    write_json(work / "features" / f"captions_{name}_human_sample.json", sample)
    unk = (tokens == vocab.stoi["<unk>"]).sum().item() / max(1, (tokens != 0).sum().item())
    print(f"[human] wrote {out} tokens={tuple(tokens.shape)} unk-rate={unk:.3f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/small.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coco = load_coco_captions(cfg)
    captioner, vocab, q_vocab, _ = load_frozen_captioner(cfg["captioner_ckpt"], device)
    for name in SPLITS:
        caption_split(cfg, name, coco, captioner, vocab, q_vocab, device)


if __name__ == "__main__":
    main()
