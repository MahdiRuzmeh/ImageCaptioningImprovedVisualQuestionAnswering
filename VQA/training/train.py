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
    score = 0.0
    for p,ans in zip(pred.tolist(), gts):
        s = " ".join([vocab.itos[i] for i in p if i < len(vocab.itos) and i > 2]).strip().lower()
        c = sum(1 for x in ans if x.strip().lower() == s)
        score += min(c/3.0, 1.0)
    return score / max(1, len(gts))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint (.pt) to resume from")
    parser.add_argument("--continue", dest="do_continue", action="store_true", help="Resume from save_dir/last.pt (or --resume if provided)")
    parser.add_argument("--fresh", action="store_true", help="Start training from scratch (ignore any resume settings)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"]=="cuda" else "cpu")
    qids = all_qids(cfg["dataset_root"])
    tr_qids, va_qids = split_qids(qids, seed=cfg["seed"])
    qv,av = build_vocabs(cfg["dataset_root"], tr_qids, cfg["vocab_min_freq"])

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
    scaler = GradScaler(enabled=cfg["use_amp"] and device.type=="cuda")
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

    for ep in range(start_epoch, cfg["epochs"]+1):
        model.train()
        tr_loss=0.0
        tr_acc=0.0
        n=0
        opt.zero_grad(set_to_none=True)
        for i,b in enumerate(tr_loader):
            images = b["images"].to(device, non_blocking=device.type == "cuda")
            q = b["q"].to(device, non_blocking=device.type == "cuda")
            a = b["a"].to(device, non_blocking=device.type == "cuda")
            with autocast(enabled=cfg["use_amp"] and device.type=="cuda"):
                logits = model(images, q, a_ids=a)
                loss = crit(logits.reshape(-1, logits.size(-1)), a[:,1:].reshape(-1))
            scaler.scale(loss / cfg["grad_accum_steps"]).backward()
            if (i+1) % cfg["grad_accum_steps"] == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            tr_loss += float(loss.item())
            tr_acc += vqa_acc(logits.argmax(dim=-1), b["answers"], av)
            n += 1

        model.eval()
        va_loss=0.0
        va_acc_score=0.0
        m=0
        with torch.no_grad():
            for b in va_loader:
                images = b["images"].to(device, non_blocking=device.type == "cuda")
                q = b["q"].to(device, non_blocking=device.type == "cuda")
                a = b["a"].to(device, non_blocking=device.type == "cuda")
                logits = model(images, q, a_ids=a)
                loss = crit(logits.reshape(-1, logits.size(-1)), a[:,1:].reshape(-1))
                va_loss += float(loss.item())
                va_acc_score += vqa_acc(logits.argmax(dim=-1), b["answers"], av)
                m += 1

        sch.step()
        tr_loss/=max(1,n); tr_acc/=max(1,n); va_loss/=max(1,m); va_acc_score/=max(1,m)
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
