"""Precompute everything frozen, once: Faster R-CNN regions + captioner outputs.

Stage 1 (regions): captioner-e RegionEncoder (hamoon detector + normalization ke
captioner bahash train shode) → raw ROI (32, 1024) har image →
``work_dir/region_cache/{train,val}/{image_id}.pt`` + yek file-e jam
``work_dir/features/regions_{split}.pt``.

Stage 2 (captions): captioner-e frozen (kaggle_mini) baraye har (image, soal):
top-``caption_beam`` beam caption + score-e length-normalized + decoder hidden
state-e beam 1 → ``work_dir/features/captions_{split}.pt``.

    python prepare_features.py --config configs/small.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from common import add_captioner_to_path, load_config, read_json, tok, write_json

add_captioner_to_path()
from models import build_captioner  # noqa: E402
from models.beam_utils import block_repeat_ngram, expand_state  # noqa: E402
from models.captioner_v1 import RegionEncoder  # noqa: E402

SPLITS = ("train", "dev", "test")


class _Itos:
    def __init__(self, itos: List[str]) -> None:
        self.itos = list(itos)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad_id = 0


def load_frozen_captioner(ckpt_path: str, device: torch.device) -> Tuple[torch.nn.Module, _Itos, _Itos, dict]:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    vocab, q_vocab = _Itos(state["vocab"]), _Itos(state["q_vocab"])
    model = build_captioner(state["config"], vocab, q_vocab)
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    # Finglish: detector weight ha (save_detector_weights=false) dar ckpt nistan → az torchvision
    bad = [k for k in missing if not k.startswith("region_encoder.detector.")]
    if bad or unexpected:
        raise RuntimeError(f"captioner ckpt mismatch: missing={bad} unexpected={unexpected}")
    for p in model.parameters():
        p.requires_grad = False
    return model.eval().to(device), vocab, q_vocab, state["config"]


def encode_question(question: str, q_vocab: _Itos, max_len: int) -> List[int]:
    unk = q_vocab.stoi.get("<unk>", 3)
    return [1] + [q_vocab.stoi.get(w, unk) for w in tok(question)[: max_len - 2]] + [2]


def pad_batch(rows: List[List[int]]) -> torch.Tensor:
    out = torch.zeros((len(rows), max(len(r) for r in rows)), dtype=torch.long)
    for i, r in enumerate(rows):
        out[i, : len(r)] = torch.tensor(r)
    return out


# ---------------------------------------------------------------------------
# Stage 1 — regions
# ---------------------------------------------------------------------------
class _ImageDS(Dataset):
    def __init__(self, ids: List[int], images_dir: str, template: str, size: int) -> None:
        self.ids, self.dir, self.template = ids, Path(images_dir), template
        self.tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        iid = self.ids[i]
        img = Image.open(self.dir / self.template.format(image_id=iid)).convert("RGB")
        return self.tf(img), iid


def image_split_of(work: Path, name: str) -> str:
    """Finglish: image folder (train/val) az ``image_split``-e record ha (yek split = yek folder)."""
    kinds = {r["image_split"] for r in read_json(work / "split" / f"{name}.json")}
    assert len(kinds) == 1, f"{name}: mixed image splits {kinds}"
    return kinds.pop()


def cache_regions(cfg: dict, device: torch.device, splits: List[str]) -> None:
    work = Path(cfg["work_dir"])
    encoder = RegionEncoder(int(cfg["max_regions"]), 2048).to(device).eval()
    for img_split, template in (("train", "COCO_train2014_{image_id:012d}.jpg"),
                                ("val", "COCO_val2014_{image_id:012d}.jpg")):
        names = [n for n in splits if image_split_of(work, n) == img_split]
        if not names:
            continue
        ids = sorted({r["image_id"] for n in names for r in read_json(work / "split" / f"{n}.json")})
        cache_dir = work / "region_cache" / img_split
        todo = [i for i in ids if not (cache_dir / f"{i}.pt").exists()]
        print(f"[regions] {img_split}: {len(ids)} images, {len(todo)} to extract")
        if todo:
            ds = _ImageDS(todo, cfg[f"{img_split}_images_dir"], template, int(cfg["image_size"]))
            dl = DataLoader(ds, batch_size=8, num_workers=6, pin_memory=True)
            with torch.no_grad():
                for imgs, iids in tqdm(dl, desc=f"regions/{img_split}"):
                    encoder(imgs.to(device), image_ids=iids, cache_dir=str(cache_dir), save_cache=True)
    # Finglish: yek file-e jam baraye har split (training ba RAM, bedoon disk I/O)
    for name in splits:
        out = work / "features" / f"regions_{name}.pt"
        if out.exists():
            continue
        img_split = image_split_of(work, name)
        ids = sorted({r["image_id"] for r in read_json(work / "split" / f"{name}.json")})
        feats = torch.stack([
            torch.load(work / "region_cache" / img_split / f"{i}.pt", map_location="cpu").half()
            for i in ids
        ])
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"image_ids": ids, "feats": feats}, out)
        print(f"[regions] wrote {out} {tuple(feats.shape)}")


# ---------------------------------------------------------------------------
# Stage 2 — captions
# ---------------------------------------------------------------------------
@torch.no_grad()
def beam_search_all(step_fn, state, n: int, device: torch.device, max_len: int, beam: int,
                    length_alpha: float = 0.7, no_repeat_ngram: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mesl ``models.beam_utils.beam_search`` vali HAME beam ha (sorted) + score bar migardoone.

    Returns: seqs (n, beam, T) ba BOS aval; norm_scores (n, beam) (length-normalized log-prob).
    """
    state = expand_state(state, n, beam)
    beam_scores = torch.full((n, beam), float("-inf"), device=device)
    beam_scores[:, 0] = 0.0
    beam_scores = beam_scores.reshape(n * beam)
    seqs = torch.full((n * beam, 1), 1, dtype=torch.long, device=device)
    finished = torch.zeros(n * beam, dtype=torch.bool, device=device)
    for _ in range(max_len - 1):
        logp, state = step_fn(seqs, state)
        logp = logp.float().clone()
        vocab = logp.size(-1)
        if no_repeat_ngram > 0:
            block_repeat_ngram(logp, seqs, no_repeat_ngram)
        if bool(finished.any()):
            logp[finished] = float("-inf")
            logp[finished, 0] = 0.0
        total = (beam_scores.unsqueeze(-1) + logp).view(n, beam * vocab)
        top_scores, top_idx = total.topk(beam, dim=-1)
        beam_id = torch.div(top_idx, vocab, rounding_mode="floor")
        token_id = top_idx % vocab
        flat = (beam_id + (torch.arange(n, device=device) * beam).unsqueeze(-1)).reshape(n * beam)
        state = {k: v[flat] for k, v in state.items()}
        seqs, finished = seqs[flat], finished[flat]
        beam_scores = top_scores.reshape(n * beam)
        tok_ = token_id.reshape(n * beam)
        seqs = torch.cat([seqs, tok_.unsqueeze(1)], dim=1)
        finished = finished | (tok_ == 2)
        if bool(finished.all()):
            break
    lengths = (seqs != 0).sum(dim=1).clamp(min=1).float()
    norm = (beam_scores / lengths.pow(length_alpha)).view(n, beam)
    order = norm.argsort(dim=-1, descending=True)
    seqs = seqs.view(n, beam, -1).gather(1, order.unsqueeze(-1).expand(-1, -1, seqs.size(1)))
    return seqs, norm.gather(1, order)


def strip_bos_eos(seqs: torch.Tensor, max_tokens: int) -> torch.Tensor:
    """(…, T) ba BOS → (…, max_tokens) token haye caption (EOS va ba'd = 0)."""
    body = seqs[..., 1:]
    after = (body == 2).long().cumsum(-1) > 0
    body = body.masked_fill(after, 0)
    if body.size(-1) < max_tokens:
        body = torch.nn.functional.pad(body, (0, max_tokens - body.size(-1)))
    return body[..., :max_tokens]


def caption_split(cfg: dict, name: str, captioner, vocab: _Itos, q_vocab: _Itos, device) -> None:
    work = Path(cfg["work_dir"])
    out = work / "features" / f"captions_{name}.pt"
    if out.exists():
        print(f"[captions] {out} exists, skip")
        return
    recs = read_json(work / "split" / f"{name}.json")
    cache_dir = str(work / "region_cache" / image_split_of(work, name))
    beam, max_len = int(cfg["caption_beam"]), int(cfg["caption_max_len"])
    L = max_len - 1
    all_tok, all_score, all_hid = [], [], []
    bs = 128
    for i in tqdm(range(0, len(recs), bs), desc=f"captions/{name}"):
        chunk = recs[i : i + bs]
        q = pad_batch([encode_question(r["question"], q_vocab, int(cfg["max_question_len"])) for r in chunk]).to(device)
        iids = torch.tensor([r["image_id"] for r in chunk], device=device)
        dummy = torch.zeros((len(chunk), 3, 1, 1), device=device)
        s = captioner._encode(dummy, q, iids, cache_dir, False)
        seqs, scores = beam_search_all(captioner._beam_step, s, len(chunk), device, max_len, beam)
        _, h = captioner._decode(seqs[:, 0], s)              # beam-1 decoder states
        tokens = strip_bos_eos(seqs, L)                       # (n, beam, L)
        hid = h[:, 1:]
        hid = torch.nn.functional.pad(hid, (0, 0, 0, max(0, L - hid.size(1))))[:, :L]
        hid = hid * (tokens[:, 0] != 0).unsqueeze(-1)
        all_tok.append(tokens.short().cpu())
        all_score.append(scores.float().cpu())
        all_hid.append(hid.half().cpu())
    tokens = torch.cat(all_tok)
    torch.save({
        "question_ids": [r["question_id"] for r in recs],
        "tokens": tokens,
        "scores": torch.cat(all_score),
        "hidden": torch.cat(all_hid),
    }, out)
    # Finglish: chand namune baraye check-e chashmi
    sample = []
    for k in range(0, len(recs), max(1, len(recs) // 30)):
        caps = [" ".join(vocab.itos[t] for t in row.tolist() if t > 3 or t == 3) for row in tokens[k]]
        sample.append({"question": recs[k]["question"], "answers": recs[k]["answers"][:3], "captions": caps})
    write_json(work / "features" / f"captions_{name}_sample.json", sample)
    print(f"[captions] wrote {out} tokens={tuple(tokens.shape)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/small.yaml")
    p.add_argument("--splits", default=",".join(SPLITS), help="comma-separated split names in work_dir/split/")
    args = p.parse_args()
    cfg = load_config(args.config)
    splits = [s for s in args.splits.split(",") if s]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_regions(cfg, device, splits)
    captioner, vocab, q_vocab, _ = load_frozen_captioner(cfg["captioner_ckpt"], device)
    for name in splits:
        caption_split(cfg, name, captioner, vocab, q_vocab, device)


if __name__ == "__main__":
    main()
