"""Train + evaluate caption-aware VQA models (A / B / C) on precomputed features.

    python train.py --config configs/small.yaml --model A
    python train.py --config configs/small.yaml --model C --no-caption --seed 1

Natije: ``work_dir/runs/<name>/results.json`` (dev best, test normal, test ba
caption-e shuffle shode, per answer type) + ``history.json`` + ``best.pt``.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from common import load_config, normalize_answer, read_json, soft_targets, tok, vqa_score, write_json
from models import CaptionerInitTransformerVQA, CaptionPointerVQA, GatedUpDnVQA

MAX_Q = 14
MAX_CANDS = 60
NEG_WORDS = ("no", "not", "nothing", "none", "nobody", "isn't", "aren't", "don't", "doesn't")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class Vocabs:
    def __init__(self, train_recs: List[dict], answer_min_count: int, cap_ckpt: dict) -> None:
        wc = Counter(w for r in train_recs for w in tok(r["question"]))
        self.q_itos = ["<pad>", "<unk>"] + sorted(w for w, c in wc.items() if c >= 2)
        self.q_stoi = {w: i for i, w in enumerate(self.q_itos)}
        ac = Counter(normalize_answer(Counter(r["answers"]).most_common(1)[0][0]) for r in train_recs)
        self.a_itos = sorted(a for a, c in ac.items() if c >= answer_min_count and a)
        self.a_stoi = {a: i for i, a in enumerate(self.a_itos)}
        self.cap_itos: List[str] = cap_ckpt["vocab"]

    def encode_q(self, question: str) -> List[int]:
        return [self.q_stoi.get(w, 1) for w in tok(question)[:MAX_Q]]

    def q_emb_init(self, cap_ckpt: dict) -> torch.Tensor:
        """VQA q_vocab embedding, row haye moshtarak az captioner q_emb kopi mishan."""
        w = cap_ckpt["model"]["q_emb.weight"]
        d = w.size(1)
        cap_stoi = {t: i for i, t in enumerate(cap_ckpt["q_vocab"])}
        out = torch.randn(len(self.q_itos), d) * d ** -0.5
        hit = 0
        for i, t in enumerate(self.q_itos):
            j = cap_stoi.get(t)
            if j is not None and i > 1:
                out[i] = w[j]
                hit += 1
        out[0].zero_()
        print(f"q_vocab={len(self.q_itos)} (init from captioner: {hit}), answers={len(self.a_itos)}")
        return out


class VQAFeatures:
    """Hame tensor ha dar RAM; batch = index-gather (bedoon DataLoader)."""

    def __init__(self, name: str, work: Path, vocabs: Vocabs, build_cands: bool, captions: str = "generated") -> None:
        self.recs = read_json(work / "split" / f"{name}.json")
        reg = torch.load(work / "features" / f"regions_{name}.pt")
        row = {iid: i for i, iid in enumerate(reg["image_ids"])}
        self.regions = reg["feats"]
        suffix = "" if captions == "generated" else f"_{captions}"
        cap = torch.load(work / "features" / f"captions_{name}{suffix}.pt")
        assert cap["question_ids"] == [r["question_id"] for r in self.recs]
        self.cap_tokens, self.cap_scores, self.cap_hidden = cap["tokens"], cap["scores"], cap["hidden"]
        n = len(self.recs)
        self.img = torch.tensor([row[r["image_id"]] for r in self.recs])
        self.q = torch.zeros((n, MAX_Q), dtype=torch.long)
        for i, r in enumerate(self.recs):
            ids = vocabs.encode_q(r["question"])
            self.q[i, : len(ids)] = torch.tensor(ids)
        self.targets: List[Dict[int, float]] = []
        self.soft: List[Dict[str, float]] = []
        for r in self.recs:
            st = soft_targets(r["answers"])
            self.soft.append(st)
            self.targets.append({vocabs.a_stoi[a]: s for a, s in st.items() if a in vocabs.a_stoi})
        self.answer_types = [r["answer_type"] for r in self.recs]
        self.vocabs = vocabs
        self.cand: Optional[dict] = self._build_candidates() if build_cands else None
        self.perm = torch.arange(n)  # caption source index (shuffle ablation)

    def _build_candidates(self) -> dict:
        """n-gram (n ≤ 3) haye 3 beam caption → candidate string (normalized, dedupe)."""
        n, nb, L = self.cap_tokens.shape
        itos = self.vocabs.cap_itos
        start = torch.zeros((n, MAX_CANDS), dtype=torch.long)
        length = torch.zeros_like(start)
        beam = torch.zeros_like(start)
        feats = torch.zeros((n, MAX_CANDS, 3))
        strings: List[List[str]] = []
        for i in range(n):
            found: Dict[str, Tuple[int, int, int]] = {}
            beams_with: Dict[str, set] = {}
            for b in range(nb):
                words = [itos[t] for t in self.cap_tokens[i, b].tolist() if t != 0]
                for s in range(len(words)):
                    for ln in (1, 2, 3):
                        if s + ln > len(words):
                            break
                        span = words[s : s + ln]
                        if any(w.startswith("<") for w in span):
                            continue
                        a = normalize_answer(" ".join(span))
                        if not a:
                            continue
                        beams_with.setdefault(a, set()).add(b)
                        if a not in found:
                            found[a] = (b, s, ln)
            keys = list(found)[:MAX_CANDS]
            for k, a in enumerate(keys):
                b, s, ln = found[a]
                beam[i, k], start[i, k], length[i, k] = b, s, ln
                feats[i, k] = torch.tensor([ln / 3.0, len(beams_with[a]) / nb, float(self.cap_scores[i, b])])
            strings.append(keys)
        mask = torch.zeros((n, MAX_CANDS), dtype=torch.bool)
        for i, keys in enumerate(strings):
            mask[i, : len(keys)] = True
        cov = sum(1 for i, keys in enumerate(strings) if any(self.soft[i].get(a, 0) > 0 for a in keys)) / n
        print(f"  candidates: mean {mask.sum(1).float().mean():.1f}/q, answer-in-candidates {cov:.3f}")
        return {"start": start, "len": length, "beam": beam, "feats": feats, "mask": mask, "strings": strings}

    def __len__(self) -> int:
        return len(self.recs)

    def shuffled_captions(self, seed: int) -> None:
        """Caption (va candidate) ha az soal-e yek image-e DIGE miyan."""
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
        c = self.perm[idx]
        out = {
            "q": self.q[idx],
            "regions": self.regions[self.img[idx]],
            "cap_tokens": self.cap_tokens[c],
            "cap_scores": self.cap_scores[c],
            "cap_hidden": self.cap_hidden[c],
        }
        tgt = torch.zeros((len(idx), n_answers))
        for k, i in enumerate(idx.tolist()):
            for a, s in self.targets[i].items():
                tgt[k, a] = s
        if self.cand is not None:
            for key in ("start", "len", "beam", "feats", "mask"):
                out[f"cand_{key}"] = self.cand[key][c]
            ct = torch.zeros((len(idx), MAX_CANDS))
            for k, (i, ci) in enumerate(zip(idx.tolist(), c.tolist())):
                for m, a in enumerate(self.cand["strings"][ci]):
                    ct[k, m] = self.soft[i].get(a, 0.0)
            tgt = torch.cat([tgt, ct], dim=1)
        out["target"] = tgt
        return {k: v.to(device, non_blocking=True) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Model / train / eval
# ---------------------------------------------------------------------------
def build_model(name: str, cfg: dict, vocabs: Vocabs, cap_ckpt: dict, use_caption: bool):
    K, d, p = len(vocabs.a_itos), int(cfg["hidden_dim"]), float(cfg["dropout"])
    q_init = vocabs.q_emb_init(cap_ckpt)
    word_emb = cap_ckpt["model"]["word_emb.weight"].float()
    if name == "A":
        return GatedUpDnVQA(K, q_init * math.sqrt(q_init.size(1)), word_emb, d, p, use_caption)
    if name == "B":
        return CaptionerInitTransformerVQA(K, q_init, cap_ckpt["model"], cap_ckpt["config"], p, use_caption)
    if name == "C":
        stoi = {t: i for i, t in enumerate(vocabs.cap_itos)}
        neg = torch.tensor([stoi[w] for w in NEG_WORDS if w in stoi])
        return CaptionPointerVQA(K, q_init * math.sqrt(q_init.size(1)), word_emb, neg, d, p, use_caption)
    raise ValueError(name)


def vqa_loss(logits: torch.Tensor, target: torch.Tensor, valid: Optional[torch.Tensor]) -> torch.Tensor:
    """Soft-score BCE (Teney et al. 2018), × tedad-e class (standard VQA scaling)."""
    loss = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
    if valid is not None:
        loss = loss * valid
    return loss.sum() / logits.size(0)


@torch.no_grad()
def evaluate(model, ds: VQAFeatures, vocabs: Vocabs, device, bs: int = 512, keep_preds: bool = False) -> dict:
    model.eval()
    K = len(vocabs.a_itos)
    scores: List[float] = []
    preds: List[str] = []
    from_ptr, gates = 0, []
    for s in range(0, len(ds), bs):
        idx = torch.arange(s, min(s + bs, len(ds)))
        b = ds.batch(idx, K, device)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(b)
        if getattr(model, "last_gate", None) is not None:
            gates.append(model.last_gate.float().cpu())
        elif getattr(getattr(model, "base", None), "last_gate", None) is not None:
            gates.append(model.base.last_gate.float().cpu())
        best = logits.float().argmax(-1).tolist()
        for k, (i, a) in enumerate(zip(idx.tolist(), best)):
            if a < K:
                text = vocabs.a_itos[a]
            else:
                text = ds.cand["strings"][int(ds.perm[i])][a - K]
                from_ptr += 1
            preds.append(text)
            scores.append(vqa_score(text, ds.recs[i]["answers"]))
    by_type: Dict[str, List[float]] = {}
    for t, sc in zip(ds.answer_types, scores):
        by_type.setdefault(t, []).append(sc)
    out = {"acc": 100 * sum(scores) / len(scores), "n": len(scores)}
    out.update({f"acc_{t}": 100 * sum(v) / len(v) for t, v in sorted(by_type.items())})
    if ds.cand is not None:
        out["pointer_rate"] = from_ptr / len(scores)
    if gates:
        out["gate_mean"] = float(torch.cat(gates).mean())
    if keep_preds:
        out["predictions"] = [{"question_id": r["question_id"], "question": r["question"], "pred": p,
                               "score": sc} for r, p, sc in zip(ds.recs, preds, scores)]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/small.yaml")
    p.add_argument("--model", choices=["A", "B", "C"], required=True)
    p.add_argument("--no-caption", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--captions", choices=["generated", "human", "enriched", "ft_base", "butd", "oracle"], default="generated",
                   help="human = COCO human captions (upper bound, prepare_human_captions.py); "
                        "enriched = captioner trained on COCO-context QD captions (configs/small_enriched.yaml); "
                        "ft_base = COCO-pretrained, QD fine-tuned captioner (configs/small_ft_base.yaml); "
                        "butd = BUTD-feature captioner (configs/small_butd.yaml); "
                        "oracle = gold answer + BUTD caption (cheat upper bound, prepare_oracle_captions.py)")
    args = p.parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)
    device = torch.device("cuda")
    work = Path(cfg["work_dir"])
    use_caption = not args.no_caption
    if not use_caption and args.captions != "generated":
        p.error("--captions has no effect with --no-caption")
    tag = "" if args.captions == "generated" else f"_{args.captions}"
    name = f"{args.model}{tag if use_caption else '_nocap'}_s{args.seed}"
    run_dir = work / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)

    cap_ckpt = torch.load(cfg["captioner_ckpt"], map_location="cpu", weights_only=False)
    vocabs = Vocabs(read_json(work / "split" / "train.json"), int(cfg["answer_min_count"]), cap_ckpt)
    want_c = args.model == "C" and use_caption
    print("loading features…")
    train = VQAFeatures("train", work, vocabs, want_c, args.captions)
    dev = VQAFeatures("dev", work, vocabs, want_c, args.captions)
    test = VQAFeatures("test", work, vocabs, want_c, args.captions)

    model = build_model(args.model, cfg, vocabs, cap_ckpt, use_caption).to(device)
    lr = float(cfg["learning_rate"])
    pre = model.pretrained_param_names() if hasattr(model, "pretrained_param_names") else set()
    params = [
        {"params": [q for n, q in model.named_parameters() if n not in pre], "lr": lr},
        {"params": [q for n, q in model.named_parameters() if n in pre], "lr": lr * float(cfg["pretrained_lr_scale"])},
    ]
    n_params = sum(q.numel() for q in model.parameters() if q.requires_grad)
    print(f"model {name}: {n_params/1e6:.2f}M trainable params ({sum(q.numel() for n, q in model.named_parameters() if n in pre)/1e6:.2f}M pretrained)")
    opt = torch.optim.AdamW([g for g in params if g["params"]], weight_decay=float(cfg["weight_decay"]))
    for g in opt.param_groups:
        g["base_lr"] = g["lr"]
    scaler = torch.amp.GradScaler()
    epochs = args.epochs or int(cfg["epochs"])
    bs = int(cfg["batch_size"])
    steps_per_epoch = math.ceil(len(train) / bs)
    total, warm = epochs * steps_per_epoch, int(cfg["warmup_epochs"]) * steps_per_epoch
    K = len(vocabs.a_itos)

    history, best, step = [], -1.0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        order = torch.randperm(len(train))
        run_loss = 0.0
        for s in range(0, len(train), bs):
            f = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * f
            b = train.batch(order[s : s + bs], K, device)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(b)
            valid = None
            if "cand_mask" in b:
                valid = torch.cat([torch.ones_like(b["target"][:, :K]), b["cand_mask"].float()], 1)
            loss = vqa_loss(logits, b["target"], valid)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item()
            step += 1
        train_secs = time.time() - t0
        ev = evaluate(model, dev, vocabs, device)
        rec = {"epoch": epoch, "train_loss": run_loss / steps_per_epoch, "train_secs": train_secs,
               **{f"dev_{k}": v for k, v in ev.items()}}
        history.append(rec)
        print(f"ep {epoch:2d} loss {rec['train_loss']:.3f} dev {ev['acc']:.2f} "
              f"(y/n {ev.get('acc_yes/no', 0):.1f} num {ev.get('acc_number', 0):.1f} other {ev.get('acc_other', 0):.1f})"
              f"{' ptr %.3f' % ev['pointer_rate'] if 'pointer_rate' in ev else ''}"
              f"{' gate %.3f' % ev['gate_mean'] if 'gate_mean' in ev else ''} [{train_secs:.0f}s]")
        if ev["acc"] > best:
            best = ev["acc"]
            torch.save({"model": model.state_dict(), "epoch": epoch}, run_dir / "best.pt")
        write_json(run_dir / "history.json", history)

    model.load_state_dict(torch.load(run_dir / "best.pt")["model"])
    best_epoch = torch.load(run_dir / "best.pt")["epoch"]
    res = {"name": name, "model": args.model, "use_caption": use_caption,
           "caption_source": args.captions if use_caption else None, "seed": args.seed,
           "params_M": n_params / 1e6, "best_epoch": best_epoch, "n_answers": K,
           "secs_per_epoch": sum(h["train_secs"] for h in history) / len(history),
           "dev": evaluate(model, dev, vocabs, device)}
    test_ev = evaluate(model, test, vocabs, device, keep_preds=True)
    write_json(run_dir / "test_predictions.json", test_ev.pop("predictions"))
    res["test"] = test_ev
    if use_caption:
        test.shuffled_captions(seed=123)
        res["test_shuffled_caption"] = evaluate(model, test, vocabs, device)
    write_json(run_dir / "results.json", res)
    print(f"[{name}] best ep {best_epoch} dev {res['dev']['acc']:.2f} test {res['test']['acc']:.2f}"
          + (f" shuffled-caption {res['test_shuffled_caption']['acc']:.2f}" if use_caption else ""))


if __name__ == "__main__":
    main()
