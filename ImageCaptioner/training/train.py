"""Train ``ImageCaptionerV1`` — stage one of *Image captioning improved visual question answering*.

Thesis narrative
----------------
Stage **(A)** learns to describe MSCOCO images using supervised captions; stage **(B)** freezes these
weights inside ``VQAModel`` so caption embeddings augment ROI attention. Reference PDF sections that
introduce **auxiliary caption pathway** / **pre-training**.

Optimization recipe
-------------------
Adamax + StepLR + AMP (CUDA) + gradient accumulation + CE loss ignoring PAD index ``0``.

CLI Examples
------------
::

    cd ImageCaptioner
    python training/train.py --config configs/default.yaml

Resume::

    python training/train.py --continue

Checkpoint schema
-----------------
Matches VQA side for tooling familiarity::

    {
        "epoch", "best", "model", "optimizer", "scheduler", "scaler",
        "vocab": [...], "config": {...}
    }

Examples (forward contract)
---------------------------
::

    logits = model.forward_train(images, captions, question_ids=None)
    # logits[t] predicts captions[:, t+1]

Paper tie-in
------------
Describe hyperparameters (hidden dim, regions, lr schedule) in thesis replication tables referencing
this script's YAML defaults.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.coco_caption_dataset import (
    CocoCaptionDataset,
    build_vocab,
    collate,
    select_image_ids,
)
from models.captioner_v1 import ImageCaptionerV1
from utils.common import load_config, resolve_path_fields, set_seed
from tqdm import tqdm
from torch import amp
from torch import nn
import torch
from torch.optim import Adamax
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader


def main() -> None:
    """Run stage-one caption training and write checkpoints under ``cfg["save_dir"]``.

    Flow: prepend ``ImageCaptioner`` to ``sys.path`` → load YAML and seed → resolve paths →
    caption vocab from ``train_captions_json`` → train/val ``CocoCaptionDataset`` loaders →
    ``ImageCaptionerV1`` + Adamax + StepLR + AMP (optional) + gradient accumulation → each epoch:
    teacher-forced CE on captions, validation loss, step LR, save ``last.pt`` and ``best.pt`` when
    validation loss **decreases** (lower is better; ``best`` tracks min val loss).

    Resume UX matches ``VQA/training/train.py``: ``--config``, ``--resume`` / ``--continue``,
    ``--fresh``. Mutually exclusive: ``--fresh`` vs ``--continue``.

    Raises:
        SystemExit: If both ``--fresh`` and ``--continue`` are passed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint (.pt) to resume from")
    parser.add_argument("--continue", dest="do_continue", action="store_true",
                        help="Resume from save_dir/last.pt (or --resume if provided)")
    parser.add_argument("--fresh", action="store_true",
                        help="Start training from scratch (ignore any resume settings)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    resolve_path_fields(
        cfg,
        (
            "train_captions_json",
            "val_captions_json",
            "train_images_dir",
            "val_images_dir",
            "save_dir",
        ),
    )
    if isinstance(cfg.get("resume_from"), str) and cfg["resume_from"]:
        resolve_path_fields(cfg, ("resume_from",))
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available()
                          and cfg["device"] == "cuda" else "cpu")
    tr_ids = select_image_ids(
        cfg["train_captions_json"], cfg.get("max_train_images"))
    va_ids = select_image_ids(cfg["val_captions_json"], cfg.get("max_val_images"))
    vocab = build_vocab(
        cfg["train_captions_json"], cfg["vocab_min_freq"], image_ids=tr_ids)

    tr = CocoCaptionDataset(
        cfg["train_images_dir"],
        cfg["train_captions_json"],
        vocab,
        cfg["max_caption_len"],
        cfg["train_image_filename_template"],
        image_ids=tr_ids,
    )
    va = CocoCaptionDataset(
        cfg["val_images_dir"],
        cfg["val_captions_json"],
        vocab,
        cfg["max_caption_len"],
        cfg["val_image_filename_template"],
        image_ids=va_ids,
    )
    if tr_ids is not None or va_ids is not None:
        print(
            f"Image subset: train_images={len(tr_ids) if tr_ids else 'all'} "
            f"({len(tr)} caption rows), val_images={len(va_ids) if va_ids else 'all'} "
            f"({len(va)} caption rows)")
    loader_kwargs = {
        "batch_size": cfg["batch_size"],
        "num_workers": cfg["num_workers"],
        "collate_fn": collate,
        "pin_memory": bool(cfg.get("pin_memory", False)) and device.type == "cuda",
    }
    if cfg["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = bool(
            cfg.get("persistent_workers", False))
        loader_kwargs["prefetch_factor"] = int(cfg.get("prefetch_factor", 2))

    tr_loader = DataLoader(tr, shuffle=True, **loader_kwargs)
    va_loader = DataLoader(va, shuffle=False, **loader_kwargs)

    model = ImageCaptionerV1(
        len(vocab.itos),
        vocab.pad_id,
        cfg["word_dim"],
        cfg["hidden_dim"],
        cfg["max_regions"],
        cfg["question_dim"]
    ).to(device)
    opt = Adamax(model.parameters(), lr=cfg["learning_rate"])
    sch = StepLR(opt, step_size=cfg["lr_decay_every"],
                 gamma=cfg["lr_decay_factor"])
    scaler = amp.GradScaler(
        "cuda", enabled=cfg["use_amp"] and device.type == "cuda")
    crit = nn.CrossEntropyLoss(ignore_index=0)

    best = 1e9
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
                print(
                    f"Resumed from {ckpt_path} (next_epoch={start_epoch}, best_val={best:.4f})")
            else:
                print(
                    f"Checkpoint at {ckpt_path} is not compatible; starting fresh.")
        else:
            print(
                f"Resume checkpoint not found at {ckpt_path}; starting fresh.")

    print(
        f"Starting training on device={device}, epochs={cfg['epochs']}, start_epoch={start_epoch}, train_steps={len(tr_loader)}, val_steps={len(va_loader)}")

    for ep in range(start_epoch, cfg["epochs"] + 1):
        print(f"\n[Epoch {ep}/{cfg['epochs']}]")
        model.train()
        tr_loss = 0.0
        opt.zero_grad(set_to_none=True)
        tr_pbar = tqdm(enumerate(tr_loader), total=len(
            tr_loader), desc=f"Train {ep}", leave=False)
        for i, b in tr_pbar:
            images = b["images"].to(device, non_blocking=device.type == "cuda")
            caps = b["captions"].to(device, non_blocking=device.type == "cuda")
            with amp.autocast("cuda", enabled=cfg["use_amp"] and device.type == "cuda"):
                logits = model.forward_train(images, caps)
                loss = crit(logits.reshape(-1, logits.size(-1)),
                            caps[:, 1:].reshape(-1))
            scaler.scale(loss / cfg["grad_accum_steps"]).backward()
            if (i + 1) % cfg["grad_accum_steps"] == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            tr_loss += float(loss.item())
            tr_pbar.set_postfix(
                batch_loss=f"{loss.item():.4f}", avg_loss=f"{tr_loss / (i + 1):.4f}")

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            va_pbar = tqdm(va_loader, total=len(va_loader),
                           desc=f"Val   {ep}", leave=False)
            for i, b in enumerate(va_pbar):
                images = b["images"].to(
                    device, non_blocking=device.type == "cuda")
                caps = b["captions"].to(
                    device, non_blocking=device.type == "cuda")
                logits = model.forward_train(images, caps)
                loss = crit(logits.reshape(-1, logits.size(-1)),
                            caps[:, 1:].reshape(-1))
                va_loss += float(loss.item())
                va_pbar.set_postfix(
                    batch_loss=f"{loss.item():.4f}", avg_loss=f"{va_loss / (i + 1):.4f}")

        sch.step()
        tr_loss /= max(1, len(tr_loader))
        va_loss /= max(1, len(va_loader))
        print(f"Epoch {ep}: train={tr_loss:.4f} val={va_loss:.4f}")
        state = {
            "epoch": ep,
            "best": best,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sch.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "vocab": vocab.itos,
            "config": cfg,
        }
        torch.save(state, out / "last.pt")
        if va_loss < best:
            best = va_loss
            state["best"] = best
            torch.save(state, out / "best.pt")


if __name__ == "__main__":
    main()
