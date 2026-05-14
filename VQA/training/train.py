"""End-to-end training loop for caption-augmented VQA (*thesis implementation*).

Experimental protocol (map to PDF sections on training / optimization)
-----------------------------------------------------------------------
1. Build question & answer vocabs from train split (avoid leakage).
2. Instantiate frozen caption weights via ``load_captioner`` (**stage-two** uses caption signal).
3. Optimize ``VQAModel`` with Adamax + StepLR + optional AMP + gradient accumulation.
4. Track ``vqa_acc`` (soft VQA score) alongside cross-entropy.

Paper references
----------------
- **Objective** — Combine caption-derived embeddings with attended ROI features; cite fusion equation.
- **Metrics** — ``vqa_acc`` mirrors standard min(agreement/3,1) averaging described in VQA literature.

CLI Examples
------------
Fresh run::

    cd VQA
    python training/train.py --config configs/default.yaml

Resume after interruption::

    python training/train.py --config configs/default.yaml --continue

Explicit checkpoint::

    python training/train.py --resume outputs/last.pt

Checkpoint contents
-------------------
``last.pt`` always saved; ``best.pt`` updates when validation ``vqa_acc`` improves. Dict stores
``model``, ``optimizer``, ``scheduler``, ``scaler``, vocabs, ``config`` snapshot.

Examples (metric intuition)
-----------------------------
If ground-truth answers contain ``"yes"`` three times and model predicts token sequence decoding to
``"yes"``, per-sample score ``min(3/3,1)=1``. Partial matches contribute proportionally before batch mean.
"""

import argparse
from pathlib import Path
from typing import List

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adamax
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from datasets.vqa_dataset import VQADataset, all_qids, build_vocabs, collate, split_qids
from models.captioner_adapter import load_captioner
from models.vqa_model import VQAModel
from utils.common import load_config, set_seed


def vqa_acc(pred: torch.Tensor, gts: List[List[str]], vocab: object) -> float:
    """Soft accuracy for one batch (mean over samples).

    Steps per sample:

    #. Decode predicted token ids with ``vocab.itos`` (skip specials with index ``<=2``).
    #. Count how many of the ten reference answers exactly match the decoded string (case-folded).
    #. Score ``min(count / 3, 1)``.

    Args:
        pred: Argmax ids ``(batch, answer_len)``.
        gts: Parallel list of ten raw answer strings from annotations.
        vocab: Answer ``Vocab`` instance exposing ``itos``.

    Examples:
        Suppose ``pred`` decodes to ``"cat"`` and five annotators wrote ``"cat"``::

            score_i = min(5 / 3, 1) == 1.0

    Paper reference
    ---------------
    Align with VQA v2 evaluation explanations cited in thesis experiments chapter.
    """
    score = 0.0
    for p, ans in zip(pred.tolist(), gts):
        s = " ".join([vocab.itos[i] for i in p if i < len(vocab.itos) and i > 2]).strip().lower()
        c = sum(1 for x in ans if x.strip().lower() == s)
        score += min(c / 3.0, 1.0)
    return score / max(1, len(gts))


def main() -> None:
    """Run the full VQA training pipeline and write checkpoints under ``cfg["save_dir"]``.

    Flow: load YAML and seed → build train/val question splits and vocabs → ``DataLoader``s with
    ``collate`` → frozen ``load_captioner`` + ``VQAModel`` → Adamax + StepLR + optional AMP and
    gradient accumulation → each epoch: train (CE + batch ``vqa_acc``), validate, step LR,
    save ``last.pt`` and ``best.pt`` when validation accuracy improves.

    CLI (see also module docstring): ``--config``, ``--resume`` / ``--continue``, ``--fresh``.
    Mutually exclusive: ``--fresh`` vs ``--continue``.

    Raises:
        SystemExit: If both ``--fresh`` and ``--continue`` are passed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint (.pt) to resume from")
    parser.add_argument("--continue", dest="do_continue", action="store_true", help="Resume from save_dir/last.pt (or --resume if provided)")
    parser.add_argument("--fresh", action="store_true", help="Start training from scratch (ignore any resume settings)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"] == "cuda" else "cpu")
    qids = all_qids(cfg["dataset_root"])
    tr_qids, va_qids = split_qids(qids, seed=cfg["seed"])
    qv, av = build_vocabs(cfg["dataset_root"], tr_qids, cfg["vocab_min_freq"])

    tr = VQADataset(cfg["dataset_root"], tr_qids, qv, av, cfg["max_question_len"], cfg["max_answer_len"])
    va = VQADataset(cfg["dataset_root"], va_qids, qv, av, cfg["max_question_len"], cfg["max_answer_len"])
    loader_kwargs = {
        "batch_size": cfg["batch_size"],
        "num_workers": cfg["num_workers"],
        "collate_fn": collate,
        "pin_memory": bool(cfg.get("pin_memory", False)) and device.type == "cuda",
    }
    if cfg["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = bool(cfg.get("persistent_workers", False))
        loader_kwargs["prefetch_factor"] = int(cfg.get("prefetch_factor", 2))

    tr_loader = DataLoader(tr, shuffle=True, **loader_kwargs)
    va_loader = DataLoader(va, shuffle=False, **loader_kwargs)

    captioner = load_captioner(cfg, len(qv.itos), qv.pad_id, device)
    model = VQAModel(len(qv.itos), len(av.itos), qv.pad_id, captioner, cfg["word_dim"], cfg["hidden_dim"], cfg["question_dim"], cfg["max_regions"], cfg["fuse_mode"]).to(device)

    opt = Adamax(model.parameters(), lr=cfg["learning_rate"])
    sch = StepLR(opt, step_size=cfg["lr_decay_every"], gamma=cfg["lr_decay_factor"])
    scaler = GradScaler(enabled=cfg["use_amp"] and device.type == "cuda")
    crit = nn.CrossEntropyLoss(ignore_index=0)

    best = 0.0
    start_epoch = 1
    out = Path(cfg["save_dir"])
    out.mkdir(parents=True, exist_ok=True)

    if args.fresh and args.do_continue:
        raise SystemExit("Choose only one: --fresh or --continue")

    resume_path = None
    if not args.fresh:
        resume_path = args.resume
        if resume_path is None and args.do_continue:
            resume_path = str(out / "last.pt")
        if resume_path is None:
            resume_path = cfg.get("resume_from")

    if resume_path:
        ckpt_path = Path(resume_path)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            if isinstance(ckpt, dict) and "model" in ckpt:
                model.load_state_dict(ckpt["model"])
                if "optimizer" in ckpt:
                    opt.load_state_dict(ckpt["optimizer"])
                if "scheduler" in ckpt:
                    sch.load_state_dict(ckpt["scheduler"])
                if "scaler" in ckpt and scaler is not None:
                    try:
                        scaler.load_state_dict(ckpt["scaler"])
                    except Exception:
                        pass
                if "best" in ckpt:
                    best = float(ckpt["best"])
                if "epoch" in ckpt:
                    start_epoch = int(ckpt["epoch"]) + 1
                print(f"Resumed from {ckpt_path} (next_epoch={start_epoch}, best_val_acc={best:.4f})")
            else:
                print(f"Checkpoint at {ckpt_path} is not compatible; starting fresh.")
        else:
            print(f"Resume checkpoint not found at {ckpt_path}; starting fresh.")

    for ep in range(start_epoch, cfg["epochs"] + 1):
        model.train()
        tr_loss = 0.0
        tr_acc = 0.0
        n = 0
        opt.zero_grad(set_to_none=True)
        for i, b in enumerate(tr_loader):
            images = b["images"].to(device, non_blocking=device.type == "cuda")
            q = b["q"].to(device, non_blocking=device.type == "cuda")
            a = b["a"].to(device, non_blocking=device.type == "cuda")
            with autocast(enabled=cfg["use_amp"] and device.type == "cuda"):
                logits = model(images, q, a_ids=a)
                loss = crit(logits.reshape(-1, logits.size(-1)), a[:, 1:].reshape(-1))
            scaler.scale(loss / cfg["grad_accum_steps"]).backward()
            if (i + 1) % cfg["grad_accum_steps"] == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            tr_loss += float(loss.item())
            tr_acc += vqa_acc(logits.argmax(dim=-1), b["answers"], av)
            n += 1

        model.eval()
        va_loss = 0.0
        va_acc_score = 0.0
        m = 0
        with torch.no_grad():
            for b in va_loader:
                images = b["images"].to(device, non_blocking=device.type == "cuda")
                q = b["q"].to(device, non_blocking=device.type == "cuda")
                a = b["a"].to(device, non_blocking=device.type == "cuda")
                logits = model(images, q, a_ids=a)
                loss = crit(logits.reshape(-1, logits.size(-1)), a[:, 1:].reshape(-1))
                va_loss += float(loss.item())
                va_acc_score += vqa_acc(logits.argmax(dim=-1), b["answers"], av)
                m += 1

        sch.step()
        tr_loss /= max(1, n)
        tr_acc /= max(1, n)
        va_loss /= max(1, m)
        va_acc_score /= max(1, m)
        print(f"Epoch {ep}: train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} val_loss={va_loss:.4f} val_acc={va_acc_score:.4f}")
        state = {
            "epoch": ep,
            "best": best,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sch.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "q_vocab": qv.itos,
            "a_vocab": av.itos,
            "config": cfg,
        }
        torch.save(state, out / "last.pt")
        if va_acc_score > best:
            best = va_acc_score
            state["best"] = best
            torch.save(state, out / "best.pt")


if __name__ == "__main__":
    main()
