"""Train + evaluate the VQA baseline (no caption) or the same model with BLIP captions.

    python train_vqa.py --config configs/blip.yaml --seed 0                       # baseline
    python train_vqa.py --config configs/blip.yaml --seed 0 --captions blip       # + BLIP captions
    python train_vqa.py --config configs/blip.yaml --seed 0 --captions oracle     # ceiling (gold answer in caption)

Natije: work_dir/runs/<captions>_s<seed>/
    results.json          dev (best epoch) + test: overall, per answer type (yes/no, number, other),
                          per question type (65 VQA v2 types), gate; + test with shuffled captions
    history.json          per-epoch dev scores
    test_predictions.json question, caption, prediction, VQA score
    best.pt
Summary of all runs: python summarize_vqa.py --config configs/blip.yaml
"""

from __future__ import annotations

import argparse
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from common import load_config, normalize_answer, read_json, soft_targets, tok, vqa_score, write_json
from models_vqa import UpDnVQA


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_captions(cfg: dict) -> Optional[Dict[int, str]]:
    p = Path(cfg["work_dir"]) / "captions" / "captions_blip.json"
    if not p.exists():
        return None
    return {int(k): v for k, v in read_json(p).items()}


def majority_answer(answers: List[str]) -> str:
    return Counter(normalize_answer(a) for a in answers).most_common(1)[0][0]


def caption_text(source: str, rec: dict, blip: Optional[Dict[int, str]]) -> str:
    if source == "none":
        return ""
    cap = blip[rec["question_id"]]
    if source == "oracle":
        # Finglish: cheat upper bound — javab-e dorost (majority-e 10 nafar) jolo-ye caption-e BLIP
        return f"the answer is {majority_answer(rec['answers'])} . {cap}"
    return cap


class Vocab:
    """Finglish: yek jadval-e kalame baraye soal + caption (az train). Aval kalamat-e soal (hamishe
    yeksan), ba'd kalamat-e faghat-caption → baseline (bedoon caption, notebook-e joda) hamoon index
    haye soal ro dare; row haye caption ro estefade nemikone."""

    def __init__(self, train: List[dict], blip: Optional[Dict[int, str]], min_count: int, answer_min: int,
                 with_captions: bool) -> None:
        qc = Counter(w for r in train for w in tok(r["question"]))
        q_words = sorted(w for w, c in qc.items() if c >= min_count)
        cap_words: List[str] = []
        if with_captions and blip is not None:
            cc = Counter(w for r in train for w in tok(blip[r["question_id"]]))
            seen = set(q_words)
            cap_words = sorted(w for w, c in (qc + cc).items() if c >= min_count and w not in seen)
        self.itos = ["<pad>", "<unk>"] + q_words + cap_words
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        ac = Counter(majority_answer(r["answers"]) for r in train)
        self.answers = sorted(a for a, c in ac.items() if c >= answer_min and a)
        self.a_stoi = {a: i for i, a in enumerate(self.answers)}

    def encode(self, text: str, max_len: int) -> List[int]:
        return [self.stoi.get(w, 1) for w in tok(text)[:max_len]]


class Split:
    """Finglish: hame tensor ha dar RAM; batch = index-gather."""

    def __init__(self, name: str, cfg: dict, vocab: Vocab, source: str, blip: Optional[Dict[int, str]]) -> None:
        work = Path(cfg["work_dir"])
        self.recs = read_json(work / "split" / f"{name}.json")
        reg = torch.load(work / "features" / f"regions_{name}.pt")
        row = {iid: i for i, iid in enumerate(reg["image_ids"])}
        self.regions = reg["feats"]
        n, lq, lc = len(self.recs), int(cfg["max_question_len"]), int(cfg["max_caption_len"])
        self.img = torch.tensor([row[r["image_id"]] for r in self.recs])
        self.q = torch.zeros((n, lq), dtype=torch.long)
        self.cap = torch.zeros((n, lc), dtype=torch.long)
        self.cap_text: List[str] = []
        for i, r in enumerate(self.recs):
            ids = vocab.encode(r["question"], lq)
            self.q[i, : len(ids)] = torch.tensor(ids)
            t = caption_text(source, r, blip)
            self.cap_text.append(t)
            ids = vocab.encode(t, lc)
            if ids:
                self.cap[i, : len(ids)] = torch.tensor(ids)
        self.targets = [{vocab.a_stoi[a]: s for a, s in soft_targets(r["answers"]).items() if a in vocab.a_stoi}
                        for r in self.recs]
        self.perm = torch.arange(n)  # caption source row (shuffle diagnostic)

    def __len__(self) -> int:
        return len(self.recs)

    def shuffle_captions(self, seed: int) -> None:
        """Finglish: har soal caption-e soal-e yek tasvir-e DIGE ro migire."""
        g = random.Random(seed)
        n = len(self)
        perm = list(range(n))
        g.shuffle(perm)
        imgs = self.img.tolist()
        for i in range(n):
            while imgs[perm[i]] == imgs[i]:
                j = g.randrange(n)
                perm[i], perm[j] = perm[j], perm[i]
        self.perm = torch.tensor(perm)

    def batch(self, idx: torch.Tensor, n_answers: int, device: torch.device) -> dict:
        tgt = torch.zeros((len(idx), n_answers))
        for k, i in enumerate(idx.tolist()):
            for a, s in self.targets[i].items():
                tgt[k, a] = s
        out = {"q": self.q[idx], "regions": self.regions[self.img[idx]], "cap": self.cap[self.perm[idx]],
               "target": tgt}
        return {k: v.to(device, non_blocking=True) for k, v in out.items()}


def mean(xs: List[float]) -> float:
    return 100 * sum(xs) / max(1, len(xs))


@torch.no_grad()
def evaluate(model, ds: Split, vocab: Vocab, device, keep_preds: bool = False, bs: int = 512) -> dict:
    model.eval()
    K = len(vocab.answers)
    scores, preds, gates = [], [], []
    for s in range(0, len(ds), bs):
        idx = torch.arange(s, min(s + bs, len(ds)))
        b = ds.batch(idx, K, device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(b)
        if model.last_gate is not None:
            gates.append(model.last_gate.float().cpu())
        for i, a in zip(idx.tolist(), logits.float().argmax(-1).tolist()):
            preds.append(vocab.answers[a])
            scores.append(vqa_score(vocab.answers[a], ds.recs[i]["answers"]))
    by_a, by_q = defaultdict(list), defaultdict(list)
    for r, sc in zip(ds.recs, scores):
        by_a[r["answer_type"]].append(sc)
        by_q[r["question_type"]].append(sc)
    out = {"acc": mean(scores), "n": len(scores),
           "answer_type": {t: {"acc": mean(v), "n": len(v)} for t, v in sorted(by_a.items())},
           "question_type": {t: {"acc": mean(v), "n": len(v)} for t, v in sorted(by_q.items())}}
    if gates:
        out["gate_mean"] = float(torch.cat(gates).mean())  # 1 = only image, 0 = only caption
    if keep_preds:
        out["predictions"] = [{"question_id": r["question_id"], "question": r["question"],
                               "answer_type": r["answer_type"], "question_type": r["question_type"],
                               "caption": ds.cap_text[int(ds.perm[i])], "pred": p,
                               "gt": majority_answer(r["answers"]), "score": sc}
                              for i, (r, p, sc) in enumerate(zip(ds.recs, preds, scores))]
    return out


def fmt(ev: dict) -> str:
    a = ev["answer_type"]
    return (f"{ev['acc']:.2f} (y/n {a.get('yes/no', {}).get('acc', 0):.1f} num {a.get('number', {}).get('acc', 0):.1f} "
            f"other {a.get('other', {}).get('acc', 0):.1f})" + (f" gate {ev['gate_mean']:.3f}" if "gate_mean" in ev else ""))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/blip.yaml")
    p.add_argument("--captions", choices=["none", "blip", "oracle"], default="none")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    work = Path(cfg["work_dir"])
    name = f"{'nocap' if args.captions == 'none' else args.captions}_s{args.seed}"
    run_dir = work / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)

    blip = load_captions(cfg)
    if args.captions != "none" and blip is None:
        p.error(f"{work / 'captions' / 'captions_blip.json'} not found (generate it on Kaggle first)")
    train_recs = read_json(work / "split" / "train.json")
    vocab = Vocab(train_recs, blip, int(cfg["word_min_count"]), int(cfg["answer_min_count"]),
                  with_captions=args.captions != "none")
    print(f"[{name}] vocab {len(vocab.itos)} words (caption words added: {args.captions != 'none'}), "
          f"{len(vocab.answers)} answers")
    train, dev, test = (Split(s, cfg, vocab, args.captions, blip) for s in ("train", "dev", "test"))
    use_caption = args.captions != "none"
    model = UpDnVQA(len(vocab.itos), len(vocab.answers), int(cfg["word_dim"]), int(cfg["hidden_dim"]),
                    int(cfg["region_dim"]), float(cfg["dropout"]), use_caption).to(device)
    n_params = sum(q.numel() for q in model.parameters() if q.requires_grad)
    print(f"[{name}] {n_params / 1e6:.2f}M params, train {len(train)} / dev {len(dev)} / test {len(test)}")

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    epochs = args.epochs or int(cfg["epochs"])
    bs = int(cfg["batch_size"])
    spe = math.ceil(len(train) / bs)
    total, warm = epochs * spe, int(cfg["warmup_epochs"]) * spe
    base_lr, K = float(cfg["learning_rate"]), len(vocab.answers)
    history, best, step = [], -1.0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        t0, run_loss = time.time(), 0.0
        order = torch.randperm(len(train))
        for s in range(0, len(train), bs):
            f = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
            for g in opt.param_groups:
                g["lr"] = base_lr * f
            b = train.batch(order[s : s + bs], K, device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(b)
            # Finglish: soft-score BCE (Teney et al. 2018), × tedad-e javab (scaling-e standard)
            loss = F.binary_cross_entropy_with_logits(logits.float(), b["target"], reduction="sum") / logits.size(0)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item()
            step += 1
        ev = evaluate(model, dev, vocab, device)
        history.append({"epoch": epoch, "train_loss": run_loss / spe, "secs": time.time() - t0, "dev": ev})
        print(f"ep {epoch:2d} loss {run_loss / spe:.3f} dev {fmt(ev)} [{time.time() - t0:.0f}s]", flush=True)
        if ev["acc"] > best:
            best = ev["acc"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "vocab": vocab.itos, "answers": vocab.answers},
                       run_dir / "best.pt")
        write_json(run_dir / "history.json", history)

    state = torch.load(run_dir / "best.pt")
    model.load_state_dict(state["model"])
    res = {"name": name, "captions": args.captions, "seed": args.seed, "params_M": n_params / 1e6,
           "best_epoch": state["epoch"], "n_answers": K, "vocab_size": len(vocab.itos),
           "train_questions": len(train), "dev": evaluate(model, dev, vocab, device)}
    t = evaluate(model, test, vocab, device, keep_preds=True)
    write_json(run_dir / "test_predictions.json", t.pop("predictions"))
    res["test"] = t
    if use_caption:
        test.shuffle_captions(seed=123)
        res["test_shuffled_captions"] = evaluate(model, test, vocab, device)
    write_json(run_dir / "results.json", res)
    print(f"[{name}] best epoch {state['epoch']}  dev {fmt(res['dev'])}")
    print(f"[{name}] TEST {fmt(t)}")
    if use_caption:
        print(f"[{name}] TEST with shuffled captions {fmt(res['test_shuffled_captions'])}")


if __name__ == "__main__":
    main()
