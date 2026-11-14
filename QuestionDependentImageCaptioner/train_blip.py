"""Model B — BLIP (pretrained ViT-B/16 + BERT text decoder) fine-tuned as a question-dependent captioner.

Finglish — idea:
    BLIP captioning (``Salesforce/blip-image-captioning-base``) decoder-e BERT-e ke dar har layer
    cross-attention be patch token haye ViT dare. Soal ro be onvan-e **prompt** jolo-ye caption
    mizarim:  [DEC] question: <soal> caption: <caption> [SEP]
    → self-attention-e decoder soal ro mibine, cross-attention tasvir ro → har layer rabete-ye
    soal-tasvir ro yad migire. Loss faghat roye token haye caption (prompt = -100).

    - Vision encoder freeze (``freeze_vision: true``): sari tar va ba 75k image overfit nemishe.
    - Data/split/metric = hamoon model A (``load_qd_json`` filter-e ok, ``*_row_range``, metrics.py),
      khorooji ``test_metrics.json`` hamoon format-e benchmark.py → scripts/results_table.py.
    - DDP ba torchrun (2x T4), resume ba ``--continue`` (last.pt har epoch), ``max_hours`` →
      ghabl az limit-e 12h-e Kaggle tamiz stop mikone.

    torchrun --nproc_per_node=2 train_blip.py --config configs/b_blip_full.yaml --continue
    python train_blip.py --config configs/b_blip_full.yaml --test --beam 3
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tqdm import tqdm

from benchmark import score_block
from train import PROJECT_ROOT, assert_disjoint_splits, load_config, resolve_path_fields, set_seed, split_rows, tok

# Finglish: BlipImageProcessor (blip-image-captioning-base): 384x384 bicubic + CLIP mean/std
BLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
BLIP_STD = (0.26862954, 0.26130258, 0.27577711)


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------
def ddp_setup(cfg: Dict[str, Any]) -> Tuple[bool, int, int]:
    if not cfg.get("ddp", False) or "RANK" not in os.environ:
        return False, 0, 1
    dist.init_process_group(backend=str(cfg.get("ddp_backend", "nccl")))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    return True, dist.get_rank(), dist.get_world_size()


def log(rank: int, *a: Any) -> None:
    if rank == 0:
        print(*a, flush=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class QdBlipDataset(Dataset):
    """(image, prompt ids, caption ids) — ids bedoon padding; collate pad mikone."""

    def __init__(self, rows: List[Dict[str, Any]], cfg: Dict[str, Any], tokenizer, bos_id: int, sep_id: int) -> None:
        self.rows = rows
        self.dir = Path(cfg["train_images_dir"])
        self.template = cfg["train_image_filename_template"]
        self.tf = transforms.Compose([
            transforms.Resize((int(cfg.get("image_size", 384)),) * 2,
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(BLIP_MEAN, BLIP_STD),
        ])
        self.tokenizer, self.bos, self.sep = tokenizer, bos_id, sep_id
        self.prompt_fmt = str(cfg.get("prompt", "question: {question} caption:"))
        self.max_q = int(cfg.get("max_prompt_tokens", 32))
        self.max_c = int(cfg.get("max_caption_tokens", 32))
        # Finglish: prompt ids yek bar hesab mishan (baraye bucket-e tool dar generate ham lazem)
        self.prompt_ids = [
            [self.bos] + tokenizer.encode(self.prompt_fmt.format(question=r["question"]),
                                          add_special_tokens=False)[: self.max_q]
            for r in rows
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        r = self.rows[i]
        img = Image.open(self.dir / self.template.format(image_id=int(r["image_id"]))).convert("RGB")
        cap = self.tokenizer.encode(str(r["caption"]), add_special_tokens=False)[: self.max_c - 1] + [self.sep]
        return {"pixel_values": self.tf(img), "prompt": self.prompt_ids[i], "caption": cap, "idx": i}


def collate_train(batch: List[Dict[str, Any]], pad_id: int) -> Dict[str, torch.Tensor]:
    """Right padding: [prompt ; caption ; PAD]; labels = -100 roye prompt va PAD."""
    seqs = [b["prompt"] + b["caption"] for b in batch]
    t = max(len(s) for s in seqs)
    ids = torch.full((len(batch), t), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), t), dtype=torch.long)
    labels = torch.full((len(batch), t), -100, dtype=torch.long)
    for k, (b, s) in enumerate(zip(batch, seqs)):
        ids[k, : len(s)] = torch.tensor(s)
        mask[k, : len(s)] = 1
        p = len(b["prompt"])
        labels[k, p : len(s)] = torch.tensor(b["caption"])
    return {"pixel_values": torch.stack([b["pixel_values"] for b in batch]), "input_ids": ids,
            "attention_mask": mask, "labels": labels, "idx": torch.tensor([b["idx"] for b in batch])}


def collate_gen(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Generate: hame prompt ha ye batch hamtool hastan (LengthBucketSampler) → padding nadare."""
    assert len({len(b["prompt"]) for b in batch}) == 1, "generation batch must share one prompt length"
    return {"pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "input_ids": torch.tensor([b["prompt"] for b in batch]),
            "idx": torch.tensor([b["idx"] for b in batch])}


class LengthBucketSampler(Sampler[List[int]]):
    """Finglish: batch haye ba prompt-e hamtool (BLIP generate ba padding-e chap/rast kharab mishe).

    Har rank DDP shard-e khodesh (indices[rank::world]) ro migire.
    """

    def __init__(self, prompt_lens: List[int], batch_size: int, rank: int = 0, world: int = 1) -> None:
        groups: Dict[int, List[int]] = defaultdict(list)
        for i in range(rank, len(prompt_lens), world):
            groups[prompt_lens[i]].append(i)
        self.batches = [idx[k : k + batch_size] for _, idx in sorted(groups.items())
                        for k in range(0, len(idx), batch_size)]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class BlipQD(nn.Module):
    """Wrapper: vision (optional freeze) → text decoder ba prompt-e soal."""

    def __init__(self, blip, freeze_vision: bool) -> None:
        super().__init__()
        self.blip = blip
        self.freeze_vision = freeze_vision
        if freeze_vision:
            for p in self.blip.vision_model.parameters():
                p.requires_grad_(False)
        tc = blip.config.text_config
        self.bos, self.sep, self.pad = int(tc.bos_token_id), int(tc.sep_token_id), int(tc.pad_token_id)

    def image_embeds(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.freeze_vision:
            self.blip.vision_model.eval()
            with torch.no_grad():
                return self.blip.vision_model(pixel_values=pixel_values).last_hidden_state
        return self.blip.vision_model(pixel_values=pixel_values).last_hidden_state

    def forward(self, pixel_values, input_ids, attention_mask, labels) -> torch.Tensor:
        emb = self.image_embeds(pixel_values)
        out = self.blip.text_decoder(
            input_ids=input_ids, attention_mask=attention_mask, encoder_hidden_states=emb,
            encoder_attention_mask=torch.ones(emb.shape[:2], dtype=torch.long, device=emb.device),
            labels=labels, reduction="mean",
        )
        return out.loss

    @torch.no_grad()
    def generate(self, pixel_values, prompt_ids, beam: int, max_new_tokens: int) -> torch.Tensor:
        """→ faghat token haye caption (ba'd az prompt)."""
        emb = self.image_embeds(pixel_values)
        out = self.blip.text_decoder.generate(
            input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids),
            encoder_hidden_states=emb,
            encoder_attention_mask=torch.ones(emb.shape[:2], dtype=torch.long, device=emb.device),
            eos_token_id=self.sep, pad_token_id=self.pad, num_beams=beam,
            max_new_tokens=max_new_tokens, do_sample=False,
        )
        return out[:, prompt_ids.size(1):]


def trainable_state(model: BlipQD) -> Dict[str, torch.Tensor]:
    """Finglish: faghat param haye train shode (vision-e freeze az base model load mishe)."""
    names = {n for n, p in model.blip.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in model.blip.state_dict().items() if k in names}


def build_model(cfg: Dict[str, Any], device: torch.device, state: Optional[Dict[str, Any]] = None):
    from transformers import AutoTokenizer, BlipForConditionalGeneration

    name = str(cfg.get("blip_model", "Salesforce/blip-image-captioning-base"))
    tokenizer = AutoTokenizer.from_pretrained(name)
    blip = BlipForConditionalGeneration.from_pretrained(name)
    if cfg.get("label_smoothing") is not None:
        blip.text_decoder.label_smoothing = float(cfg["label_smoothing"])
    model = BlipQD(blip, bool(cfg.get("freeze_vision", True)))
    if state is not None:
        missing, unexpected = model.blip.load_state_dict(state, strict=False)
        assert not unexpected, f"unexpected keys in checkpoint: {unexpected[:5]}"
    return model.to(device), tokenizer


# ---------------------------------------------------------------------------
# Decode + score
# ---------------------------------------------------------------------------
def decode_split(model: BlipQD, ds: QdBlipDataset, tokenizer, cfg, device, beam: int,
                 rank: int, world: int, ddp_on: bool) -> List[str]:
    """Finglish: har rank shard-e khodesh; all_gather → list-e kamel be tartib-e ds (hame rank ha)."""
    sampler = LengthBucketSampler([len(p) for p in ds.prompt_ids], int(cfg.get("eval_batch_size", 64)), rank, world)
    dl = DataLoader(ds, batch_sampler=sampler, num_workers=int(cfg.get("num_workers", 2)), collate_fn=collate_gen)
    m = model.module if hasattr(model, "module") else model
    m.eval()
    local: Dict[int, str] = {}
    use_amp = bool(cfg.get("use_amp", True)) and device.type == "cuda"
    for b in tqdm(dl, desc=f"decode(beam={beam})", disable=rank != 0, mininterval=30):
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            out = m.generate(b["pixel_values"].to(device), b["input_ids"].to(device), beam,
                             int(cfg.get("max_caption_tokens", 32)))
        for i, seq in zip(b["idx"].tolist(), out.tolist()):
            local[i] = tokenizer.decode(seq, skip_special_tokens=True).strip()
    if ddp_on:
        parts: List[Dict[int, str]] = [None] * world  # type: ignore[list-item]
        dist.all_gather_object(parts, local)
        for p in parts:
            local.update(p)
    return [local[i] for i in range(len(ds))]


def score(preds: List[str], rows: List[Dict[str, Any]]) -> Dict[str, float]:
    return score_block([tok(p) for p in preds], rows)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def lr_lambda(cfg: Dict[str, Any], total_steps: int):
    warm = int(cfg.get("warmup_steps", 1000))

    def f(step: int) -> float:
        if step < warm:
            return (step + 1) / max(1, warm)
        prog = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))  # Finglish: cosine be 0

    return f


def save(path: Path, obj: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)  # Finglish: atomic → kill vasat-e save checkpoint ro kharab nemikone


def train(cfg: Dict[str, Any], cli: argparse.Namespace) -> None:
    ddp_on, rank, world = ddp_setup(cfg)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    set_seed(int(cfg["seed"]) + rank)
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    if rank == 0:
        assert_disjoint_splits(cfg)
        (save_dir / "run_config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")

    last = save_dir / "last.pt"
    resume = torch.load(last, map_location="cpu", weights_only=False) if cli.do_continue and last.exists() else None
    model, tokenizer = build_model(cfg, device, resume["model"] if resume else None)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    rows_tr = split_rows(cfg, "train", cfg["train_captions_json"])
    rows_va = split_rows(cfg, "val", cfg["train_captions_json"])
    ds_tr = QdBlipDataset(rows_tr, cfg, tokenizer, model.bos, model.sep)
    ds_va = QdBlipDataset(rows_va, cfg, tokenizer, model.bos, model.sep)
    sampler = DistributedSampler(ds_tr, shuffle=True, seed=int(cfg["seed"])) if ddp_on else None
    nw = int(cfg.get("num_workers", 2))
    dl = DataLoader(ds_tr, batch_size=int(cfg["batch_size"]), shuffle=sampler is None, sampler=sampler,
                    num_workers=nw, pin_memory=device.type == "cuda", drop_last=True,
                    persistent_workers=nw > 0, collate_fn=functools.partial(collate_train, pad_id=model.pad))

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg["learning_rate"]), weight_decay=float(cfg.get("weight_decay", 0.05)))
    epochs = int(cfg["epochs"])
    accum = max(1, int(cfg.get("grad_accum_steps", 1)))
    total = epochs * math.ceil(len(dl) / accum)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(cfg, total))
    use_amp = bool(cfg.get("use_amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch, best, best_epoch = 1, -1.0, None
    if resume:
        opt.load_state_dict(resume["optimizer"]); sched.load_state_dict(resume["scheduler"])
        scaler.load_state_dict(resume["scaler"])
        start_epoch, best, best_epoch = resume["epoch"] + 1, resume["best"], resume.get("best_epoch")
        log(rank, f"resumed from {last} at epoch {resume['epoch']} (best CIDEr {best:.4f})")
        del resume

    net = model
    if ddp_on:
        from torch.nn.parallel import DistributedDataParallel as DDP
        net = DDP(model, device_ids=[local_rank],
                  find_unused_parameters=bool(cfg.get("ddp_find_unused_parameters", False)))
    log(rank, f"BLIP-QD: trainable={n_train:,} train_rows={len(ds_tr)} val_rows={len(ds_va)} "
              f"world={world} batch/gpu={cfg['batch_size']} steps/epoch={len(dl)} epochs={epochs}")

    max_secs = float(cfg.get("max_hours", 1e9)) * 3600
    epoch_secs = 0.0
    for epoch in range(start_epoch, epochs + 1):
        # Finglish: epoch-e badi ja nemishe → stop (last.pt hast, session-e badi --continue)
        if time.time() - t_start + epoch_secs * 1.1 > max_secs:
            log(rank, f"time budget: stop before epoch {epoch} (resume with --continue)")
            break
        t0 = time.time()
        if sampler is not None:
            sampler.set_epoch(epoch)
        net.train()
        tot, n = 0.0, 0
        opt.zero_grad(set_to_none=True)
        bar = tqdm(dl, desc=f"epoch {epoch}", disable=rank != 0, mininterval=60)
        for step, b in enumerate(bar, 1):
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                loss = net(b["pixel_values"].to(device, non_blocking=True), b["input_ids"].to(device),
                           b["attention_mask"].to(device), b["labels"].to(device))
            scaler.scale(loss / accum).backward()
            if step % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, float(cfg.get("grad_clip", 1.0)))
                scaler.step(opt); scaler.update(); sched.step()
                opt.zero_grad(set_to_none=True)
            tot += float(loss); n += 1
            if rank == 0 and step % 200 == 0:
                bar.set_postfix(loss=f"{tot / n:.4f}", lr=f"{sched.get_last_lr()[0]:.2e}")
        train_loss = tot / max(1, n)

        preds = decode_split(net, ds_va, tokenizer, cfg, device, int(cfg.get("val_beam_size", 1)), rank, world, ddp_on)
        if rank == 0:
            sc = score(preds, rows_va)
            cider = sc["CIDEr"]
            improved = cider > best
            if improved:
                best, best_epoch = cider, epoch
            state = trainable_state(model)
            if improved:
                save(save_dir / "best.pt", {"model": state, "config": cfg, "epoch": epoch, "best_epoch": epoch,
                                            "select_metric": "cider", "model_class": "BlipQD"})
            save(last, {"model": state, "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                        "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "best_epoch": best_epoch,
                        "config": cfg})
            epoch_secs = time.time() - t0
            print(f"epoch {epoch}/{epochs} train_loss={train_loss:.4f} val_CIDEr={cider:.4f} "
                  f"val_BLEU-4={sc['BLEU-4']:.4f} val_METEOR={sc['METEOR']:.4f} val_ROUGE-L={sc['ROUGE-L']:.4f} "
                  f"val_AnswerHit={sc['AnswerHit']:.4f} ({epoch_secs / 60:.1f} min)"
                  + ("  best.pt updated" if improved else ""), flush=True)
            for p, r in list(zip(preds, rows_va))[:3]:
                print(f"  Q: {r['question']}\n    pred: {p}\n    gt:   {r['caption']}", flush=True)
        if ddp_on:
            t = torch.tensor([epoch_secs], device=device)
            dist.broadcast(t, 0)
            epoch_secs = float(t.item())
    if ddp_on:
        dist.barrier()
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Test (single GPU)
# ---------------------------------------------------------------------------
def test(cfg: Dict[str, Any], cli: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(cli.ckpt or Path(cfg["save_dir"]) / "best.pt").resolve()
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model, tokenizer = build_model(cfg, device, state["model"])
    rows = split_rows(cfg, cli.split, cfg["train_captions_json"])
    ds = QdBlipDataset(rows, cfg, tokenizer, model.bos, model.sep)
    t0 = time.time()
    preds = decode_split(model, ds, tokenizer, cfg, device, cli.beam, 0, 1, False)
    secs = time.time() - t0
    overall = score(preds, rows)
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r.get("rule") or "unknown"].append(i)
    per_rule = {k: score([preds[i] for i in v], [rows[i] for i in v]) for k, v in sorted(groups.items())}
    result = {
        "name": cli.name or Path(cfg["save_dir"]).name, "model_class": "BlipQD", "seed": cfg.get("seed"),
        "ckpt": str(ckpt_path), "best_epoch": state.get("best_epoch"), "select_metric": "cider",
        "split": cli.split, "beam": cli.beam,
        "trainable_params": sum(v.numel() for v in state["model"].values()),
        "decode_secs": secs, "items_per_sec": len(rows) / max(secs, 1e-9),
        "overall": overall, "per_rule": per_rule,
        "predictions": [{"image_id": r["image_id"], "question": r["question"], "answer": r["answer"],
                         "rule": r.get("rule"), "gt": r["caption"], "pred": p} for r, p in zip(rows, preds)],
    }
    out = Path(cli.out) if cli.out else ckpt_path.parent / f"{cli.split}_metrics.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in overall.items() if k != "n"))
    for p, r in list(zip(preds, rows))[:5]:
        print(f"  Q: {r['question']}\n    pred: {p}\n    gt:   {r['caption']}")
    print(f"saved -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Model B: BLIP question-dependent captioner")
    p.add_argument("--config", required=True)
    p.add_argument("--continue", dest="do_continue", action="store_true", help="Resume from save_dir/last.pt")
    p.add_argument("--save-dir", default=None)
    p.add_argument("--test", action="store_true", help="Decode a split with best.pt and write <split>_metrics.json")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--split", default="test", choices=("val", "test"))
    p.add_argument("--beam", type=int, default=3)
    p.add_argument("--name", default=None)
    p.add_argument("--out", default=None)
    cli = p.parse_args()
    path = Path(cli.config) if Path(cli.config).is_absolute() else PROJECT_ROOT / cli.config
    cfg = load_config(str(path))
    if cli.save_dir:
        cfg["save_dir"] = cli.save_dir
    resolve_path_fields(cfg, ("train_captions_json", "train_images_dir", "save_dir"))
    if cfg.get("blip_model") and not str(cfg["blip_model"]).startswith("Salesforce/"):
        resolve_path_fields(cfg, ("blip_model",))
    test(cfg, cli) if cli.test else train(cfg, cli)


if __name__ == "__main__":
    main()
