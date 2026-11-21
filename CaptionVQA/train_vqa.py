"""Train + evaluate the VQA baseline (no caption) or the same model with BLIP captions.

    python train_vqa.py --config configs/blip.yaml --seed 0                       # baseline
    python train_vqa.py --config configs/blip.yaml --seed 0 --captions blip       # + BLIP captions
    python train_vqa.py --config configs/blip.yaml --seed 0 --captions oracle     # ceiling (gold answer in caption)
    python train_vqa.py --config configs/blip_strong_gate.yaml --seed 0 --captions blip   # + caption gate (MCAN)

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
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from common import load_config, normalize_answer, read_json, soft_targets, tok, vqa_score, write_json
from models_vqa import MCANVQA, UpDnVQA

# Finglish: hamoon ta'rif-e "caption dorost" ke dar report ha (AnswerHit / polarity-e captioner)
sys.path.insert(1, str(Path(__file__).resolve().parent.parent / "QuestionDependentImageCaptioner"))
from metrics import answer_in_caption, caption_negated  # noqa: E402

CASES = ("open, caption right", "open, caption wrong", "yes/no, polarity right", "yes/no, polarity wrong")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_captions(cfg: dict) -> Optional[Dict[int, str]]:
    # Finglish: configs/blip_full.yaml → captions_json (cross-fitted); vagarna captions/captions_blip.json
    p = Path(cfg["captions_json"]) if cfg.get("captions_json") else Path(cfg["work_dir"]) / "captions" / "captions_blip.json"
    if not p.exists():
        return None
    return {int(k): v for k, v in read_json(p).items()}


def load_glove(path: Optional[str], itos: List[str], dim: int, name: str) -> Optional[torch.Tensor]:
    """Finglish: GloVe (text: 'word v1 … vD') → jadval-e embedding baraye vocab; kalame-ye bedoon vector → random kochik.
    Path nabashe → None (embedding random, mesl-e qabl)."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"[{name}] WARNING: GloVe file {p} not found -> random word embeddings")
        return None
    want = {w: i for i, w in enumerate(itos)}
    g = torch.Generator().manual_seed(0)
    emb = torch.randn(len(itos), dim, generator=g) * 0.1
    emb[0].zero_()
    hit = 0
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            w, _, rest = line.rstrip().partition(" ")
            i = want.get(w)
            if i is None or i == 0:
                continue
            v = rest.split(" ")
            if len(v) != dim:
                continue
            emb[i] = torch.tensor([float(x) for x in v])
            hit += 1
    print(f"[{name}] GloVe: {hit}/{len(itos) - 2} vocabulary words found in {p.name}")
    return emb


def majority_answer(answers: List[str]) -> str:
    return Counter(normalize_answer(a) for a in answers).most_common(1)[0][0]


def caption_reliable(caption: str, gt: str) -> int:
    """Finglish: label-e gate: 1 = caption javab-e dorost (majority) ro dare / polarity-e dorost, 0 = na,
    -1 = ta'rif nashode (bedoon caption, ya javab-e bedoon token). Faghat target-e training + tahlil; voroodi-e model nist."""
    c = tok(caption)
    if not c:
        return -1
    if gt in ("yes", "no"):
        return int(caption_negated(c) == (gt == "no"))
    a = tok(gt)
    return int(answer_in_caption(a, c)) if a else -1


def auc(scores: List[float], labels: List[int]) -> float:
    """Finglish: ROC AUC = ehtemal-e inke caption-e dorost r-e bozorgtar az caption-e ghalat begire (tie = 1/2)."""
    s, y = np.asarray(scores, dtype=np.float64), np.asarray(labels)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ss = s[order]
    i = 0
    while i < len(s):  # average rank baraye tie ha
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if not npos or not nneg:
        return float("nan")
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def list_mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


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
        packed = (Path(cfg["features_dir"]) if cfg.get("features_dir") else work / "features") / f"regions_{name}"
        if (packed / "meta.json").exists():
            # Finglish: packed memmap (prepare_regions.py) — az disk/page cache, beyn-e process ha share
            meta = read_json(packed / "meta.json")
            ids, t, d = meta["ids"], int(meta["tokens"]), int(meta["feat_dim"])
            done = np.memmap(packed / "done.u8", np.uint8, "r", shape=(len(ids),))
            assert int(done.sum()) == len(ids), f"{packed}: {len(ids) - int(done.sum())} images not extracted"
            self.regions_mm = np.memmap(packed / "feats.f16", np.float16, "r", shape=(len(ids), t, d))
            self.regions = None
        else:
            reg = torch.load(work / "features" / f"regions_{name}.pt")
            ids, self.regions, self.regions_mm = reg["image_ids"], reg["feats"], None
        row = {iid: i for i, iid in enumerate(ids)}
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
        # Finglish: label-e reliability-e caption (gate); -1 = bedoon label
        self.reliable = torch.tensor([caption_reliable(t, majority_answer(r["answers"])) if source == "blip" else -1
                                      for t, r in zip(self.cap_text, self.recs)], dtype=torch.float)
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
        rows = self.img[idx]
        if self.regions_mm is not None:
            regions = torch.from_numpy(np.ascontiguousarray(self.regions_mm[rows.numpy()]))
        else:
            regions = self.regions[rows]
        out = {"q": self.q[idx], "regions": regions, "cap": self.cap[self.perm[idx]], "target": tgt,
               "rel": self.reliable[self.perm[idx]]}
        return {k: v.to(device, non_blocking=True) for k, v in out.items()}


def mean(xs: List[float]) -> float:
    return 100 * sum(xs) / max(1, len(xs))


@torch.no_grad()
def predict_scores(model, ds: Split, n_answers: int, device, bs: int = 512) -> np.ndarray:
    """Finglish: sigmoid score-e har javab baraye har soal (N, K) fp16 — baraye ensemble."""
    model.eval()
    out = []
    for s in range(0, len(ds), bs):
        idx = torch.arange(s, min(s + bs, len(ds)))
        b = ds.batch(idx, n_answers, device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(b)
        out.append(torch.sigmoid(logits.float()).half().cpu().numpy())
    return np.concatenate(out)


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
        if getattr(model, "last_gate", None) is not None:
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
    g = torch.cat(gates).tolist() if gates else None
    if g is not None:
        # UpDn: 1 = only image; MCAN caption gate: r = caption reliability (1 = use the caption)
        out["gate_mean"] = sum(g) / len(g)
    shuffled = not torch.equal(ds.perm, torch.arange(len(ds)))
    rel = ds.reliable.tolist()
    if not shuffled and any(x >= 0 for x in rel):
        # Finglish: 4 halat: open/yes-no x caption dorost/ghalat (+ miangin-e r har halat)
        cases = defaultdict(list)
        for i, r in enumerate(ds.recs):
            if rel[i] < 0:
                continue
            yn = majority_answer(r["answers"]) in ("yes", "no")
            cases[CASES[2 * int(yn) + int(rel[i] == 0)]].append(i)
        out["cases"] = {k: {"acc": mean([scores[i] for i in cases[k]]), "n": len(cases[k]),
                            **({"r_mean": list_mean([g[i] for i in cases[k]])} if g is not None else {})}
                        for k in CASES}
        if g is not None:
            lab = [i for i in range(len(ds)) if rel[i] >= 0]
            out["gate"] = {"auc": auc([g[i] for i in lab], [int(rel[i]) for i in lab]),
                           "r_mean_reliable": list_mean([g[i] for i in lab if rel[i] == 1]),
                           "r_mean_unreliable": list_mean([g[i] for i in lab if rel[i] == 0])}
    if keep_preds:
        out["predictions"] = [{"question_id": r["question_id"], "question": r["question"],
                               "answer_type": r["answer_type"], "question_type": r["question_type"],
                               "caption": ds.cap_text[int(ds.perm[i])], "pred": p,
                               "gt": majority_answer(r["answers"]), "score": sc,
                               **({"r": round(g[i], 4), "reliable": int(rel[i])} if g is not None else {})}
                              for i, (r, p, sc) in enumerate(zip(ds.recs, preds, scores))]
    return out


def fmt(ev: dict) -> str:
    a = ev["answer_type"]
    s = (f"{ev['acc']:.2f} (y/n {a.get('yes/no', {}).get('acc', 0):.1f} num {a.get('number', {}).get('acc', 0):.1f} "
         f"other {a.get('other', {}).get('acc', 0):.1f})" + (f" gate {ev['gate_mean']:.3f}" if "gate_mean" in ev else ""))
    if "gate" in ev:
        s += f" r-AUC {ev['gate']['auc']:.3f}"
    return s


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/blip.yaml")
    p.add_argument("--captions", choices=["none", "blip", "oracle"], default="none")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default=None, help="e.g. cuda:1 (Kaggle: one seed per GPU)")
    p.add_argument("--caption-dropout", type=float, default=None,
                   help="share of training questions whose caption is hidden (default: config caption_dropout, else 0)")
    p.add_argument("--gate-aux-weight", type=float, default=None,
                   help="caption gate: weight of the reliability loss (default: config gate_aux_weight, else 0)")
    args = p.parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    work = Path(cfg["work_dir"])
    name = f"{'nocap' if args.captions == 'none' else args.captions}_s{args.seed}"
    run_dir = work / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)

    blip = load_captions(cfg)
    if args.captions != "none" and blip is None:
        p.error("captions file not found (generate it on Kaggle first)")
    train_recs = read_json(work / "split" / "train.json")
    vocab = Vocab(train_recs, blip, int(cfg["word_min_count"]), int(cfg["answer_min_count"]),
                  with_captions=args.captions != "none")
    print(f"[{name}] vocab {len(vocab.itos)} words (caption words added: {args.captions != 'none'}), "
          f"{len(vocab.answers)} answers")
    train, dev, test = (Split(s, cfg, vocab, args.captions, blip) for s in ("train", "dev", "test"))
    use_caption = args.captions != "none"
    cap_drop = float(args.caption_dropout if args.caption_dropout is not None else cfg.get("caption_dropout", 0.0))
    if use_caption and cap_drop > 0:
        print(f"[{name}] caption dropout {cap_drop}: captions hidden for that share of training questions")
    gate = use_caption and bool(cfg.get("caption_gate", False))
    aux_w = float(args.gate_aux_weight if args.gate_aux_weight is not None else cfg.get("gate_aux_weight", 0.0))
    gate_warm = int(cfg.get("gate_warmup_epochs", cfg.get("warmup_epochs", 0)))
    if gate:
        lab = train.reliable[train.reliable >= 0]
        print(f"[{name}] caption gate on (acts from epoch {gate_warm + 1}); reliability loss weight {aux_w}; "
              f"train captions reliable: {100 * lab.mean().item():.1f}% of {len(lab)}")
    emb_init = load_glove(cfg.get("glove_path"), vocab.itos, int(cfg["word_dim"]), name)
    if cfg.get("model", "updn") == "mcan":
        model = MCANVQA(len(vocab.itos), len(vocab.answers), int(cfg["word_dim"]), int(cfg["hidden_dim"]),
                        int(cfg.get("num_heads", 8)), int(cfg.get("ff_dim", 2048)), int(cfg.get("num_layers", 6)),
                        int(cfg["region_dim"]), float(cfg["dropout"]), use_caption, emb_init, gate).to(device)
    else:
        model = UpDnVQA(len(vocab.itos), len(vocab.answers), int(cfg["word_dim"]), int(cfg["hidden_dim"]),
                        int(cfg["region_dim"]), float(cfg["dropout"]), use_caption).to(device)
        if emb_init is not None:
            model.emb.weight.data.copy_(emb_init)
    n_params = sum(q.numel() for q in model.parameters() if q.requires_grad)
    print(f"[{name}] {cfg.get('model', 'updn')}: {n_params / 1e6:.2f}M params, "
          f"train {len(train)} / dev {len(dev)} / test {len(test)}")

    if cfg.get("optimizer", "adamw") == "adam":
        # Finglish: MCAN — Adam(0.9, 0.98), bedoon weight decay
        opt = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]), betas=(0.9, 0.98), eps=1e-9)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]),
                                weight_decay=float(cfg["weight_decay"]))
    decay_epochs = [int(e) for e in cfg.get("lr_decay_epochs", [])]
    decay_factor = float(cfg.get("lr_decay_factor", 0.2))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    epochs = args.epochs or int(cfg["epochs"])
    bs = int(cfg["batch_size"])
    spe = math.ceil(len(train) / bs)
    total, warm = epochs * spe, int(cfg["warmup_epochs"]) * spe
    base_lr, K = float(cfg["learning_rate"]), len(vocab.answers)
    history, best, step = [], -1.0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        if gate:
            # Finglish: gate warm-up — aval model yad migire caption ro estefade kone, ba'd gate asar mizare
            model.gate_active = epoch > gate_warm
        t0, run_loss, run_aux = time.time(), 0.0, 0.0
        order = torch.randperm(len(train))
        for s in range(0, len(train), bs):
            if step < warm:
                f = (step + 1) / warm
            elif decay_epochs:  # Finglish: MCAN — step decay (× factor dar har epoch-e lr_decay_epochs)
                f = decay_factor ** sum(epoch >= e for e in decay_epochs)
            else:
                f = 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
            for g in opt.param_groups:
                g["lr"] = base_lr * f
            b = train.batch(order[s : s + bs], K, device)
            if use_caption and cap_drop > 0:
                # Finglish: caption dropout — baraye ye sahm az soal ha caption pak mishe (hame PAD) →
                # model yad migire bedoon caption ham az tasvir javab bede va be caption kur-kurane etemad nakone
                drop = torch.rand(b["cap"].size(0), device=device) < cap_drop
                b["cap"] = b["cap"].masked_fill(drop[:, None], 0)
                b["rel"] = b["rel"].masked_fill(drop, -1.0)  # Finglish: caption pak shode -> label nadare
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(b)
            # Finglish: soft-score BCE (Teney et al. 2018), × tedad-e javab (scaling-e standard)
            loss = F.binary_cross_entropy_with_logits(logits.float(), b["target"], reduction="sum") / logits.size(0)
            if gate and aux_w > 0:
                # Finglish: reliability loss: gate yad migire caption dorost-e ya na (label az javab-e gold, faghat target)
                m = b["rel"] >= 0
                if m.any():
                    aux = F.binary_cross_entropy_with_logits(model.last_gate_logit[m], b["rel"][m])
                    loss = loss + aux_w * aux
                    run_aux += aux.item()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item()
            step += 1
        ev = evaluate(model, dev, vocab, device)
        history.append({"epoch": epoch, "train_loss": run_loss / spe, "secs": time.time() - t0, "dev": ev,
                        **({"train_aux_loss": run_aux / spe} if gate and aux_w > 0 else {})})
        aux_s = f" (reliability {run_aux / spe:.3f})" if gate and aux_w > 0 else ""
        print(f"ep {epoch:2d} loss {run_loss / spe:.3f}{aux_s} dev {fmt(ev)} [{time.time() - t0:.0f}s]", flush=True)
        if ev["acc"] > best:
            best = ev["acc"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "vocab": vocab.itos, "answers": vocab.answers},
                       run_dir / "best.pt")
        write_json(run_dir / "history.json", history)

    state = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(state["model"])
    if gate:
        model.gate_active = state["epoch"] > gate_warm
    res = {"name": name, "captions": args.captions, "seed": args.seed, "params_M": n_params / 1e6,
           "best_epoch": state["epoch"], "n_answers": K, "vocab_size": len(vocab.itos),
           "train_questions": len(train), "dev": evaluate(model, dev, vocab, device)}
    t = evaluate(model, test, vocab, device, keep_preds=True)
    write_json(run_dir / "test_predictions.json", t.pop("predictions"))
    res["test"] = t
    if use_caption:
        test.shuffle_captions(seed=123)
        res["test_shuffled_captions"] = evaluate(model, test, vocab, device)
    res["caption_dropout"] = cap_drop if use_caption else None
    res["caption_gate"] = gate
    res["gate_aux_weight"] = aux_w if gate else None
    write_json(run_dir / "results.json", res)
    # Finglish: score ha baraye ensemble (bedoon inference-e dobare); caption-e test be halat-e normal bar migarde
    test.perm = torch.arange(len(test))
    for nm, ds in (("dev", dev), ("test", test)):
        np.save(run_dir / f"{nm}_scores.npy", predict_scores(model, ds, K, device))
    write_json(run_dir / "answers.json", vocab.answers)
    print(f"[{name}] best epoch {state['epoch']}  dev {fmt(res['dev'])}")
    print(f"[{name}] TEST {fmt(t)}")
    for k, v in t.get("cases", {}).items():
        print(f"[{name}]   {k:24s} n={v['n']:5d}  acc {v['acc']:.1f}" + (f"  r {v['r_mean']:.3f}" if "r_mean" in v else ""))
    if use_caption:
        print(f"[{name}] TEST with shuffled captions {fmt(res['test_shuffled_captions'])}")


if __name__ == "__main__":
    main()
