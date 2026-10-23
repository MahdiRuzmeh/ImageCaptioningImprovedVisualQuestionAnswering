"""QD captioner — eval va inference (Stage 1 QD).

Ba ``best.pt`` mitooni:
- yek ``image_id`` (+ soal) → caption
- roye split: **token accuracy** (teacher forcing) + BLEU/CIDEr + sample ha
- QD checkpoint: ``vocab`` + ``q_vocab`` + ``config`` az hamoon ``best.pt``

Finglish: model class az ``ckpt['config']['model_class']`` (registry
``models.build_captioner``) — MMTransformerCopyCaptioner (v2 default) ya v1.
Checkpoint ghadimi bedoon ``model_class`` → v1. Benchmark kamel roye test split:
``benchmark.py``.

Run az project folder::

    python eval.py --config configs/smoke.yaml --ckpt outputs/smoke/best.pt \\
        --split val --samples 10

    python eval.py --config configs/smoke.yaml --ckpt outputs/smoke/best.pt \\
        --image-id 9 --split train --question "How many cookies can be seen?"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from metrics import compute_caption_metrics
from models import DEFAULT_MODEL_CLASS, build_captioner
from train import (
    PROJECT_ROOT,
    Vocab,
    build_split_dataset,
    collate_batch,
    eval_epoch,
    image_cap,
    load_config,
    load_qd_json,
    region_cache_dir_for_split,
    resolve_path_fields,
    set_seed,
    tok,
)

PATH_KEYS = (
    "train_captions_json",
    "val_captions_json",
    "train_images_dir",
    "val_images_dir",
    "save_dir",
    "train_region_cache_dir",
    "val_region_cache_dir",
    "test_region_cache_dir",
)


def image_size_from_cfg(cfg: Dict[str, Any]) -> int:
    """Finglish: image_size ro az YAML migirim (default 448) ta ba train yeksan bashe."""
    return int(cfg.get("image_size", 448))


def image_transform(image_size: int) -> transforms.Compose:
    """Hamoon transform train: Resize(image_size) + ImageNet normalize."""
    size = int(image_size)
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def vocab_from_itos(itos: List[str]) -> Vocab:
    """``Vocab`` ro az list ``itos`` toye checkpoint dobare besaz."""
    v = Vocab.__new__(Vocab)
    v.itos = list(itos)
    v.stoi = {w: i for i, w in enumerate(v.itos)}
    return v


def decode_ids(ids: List[int], vocab: Vocab) -> str:
    """Token id ha ro be matn tabdil kon (PAD/BOS/EOS skip)."""
    words = [
        vocab.itos[i]
        for i in ids
        if 0 < i < len(vocab.itos)
        and vocab.itos[i] not in (vocab.PAD, vocab.BOS, vocab.EOS)
    ]
    return " ".join(words).strip()


def load_image(
    image_id: int,
    images_dir: str,
    filename_template: str,
    device: torch.device,
    image_size: int,
) -> torch.Tensor:
    """Yek COCO image ro load kon → (1, 3, H, W)."""
    path = image_path(image_id, images_dir, filename_template)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    tensor = image_transform(image_size)(Image.open(path).convert("RGB"))
    return tensor.unsqueeze(0).to(device)


def image_path(image_id: int, images_dir: str, filename_template: str) -> Path:
    """Image path for a COCO image_id (based on config template)."""
    return Path(images_dir) / filename_template.format(image_id=image_id)


def encode_question(
    question: str, q_vocab: Vocab, max_len: int, device: torch.device
) -> torch.Tensor:
    """Matn soal → BOS + tokens + EOS → tensor (1, seq_len)."""
    ids = [1] + q_vocab.encode(tok(question)[: max_len - 2]) + [2]
    return torch.tensor([ids], dtype=torch.long, device=device)


def build_model_from_ckpt(
    ckpt_path: Path,
    cfg: Dict[str, Any],
    device: torch.device,
    vqa_ckpt: Optional[Path] = None,
) -> Tuple[nn.Module, Vocab, Vocab]:
    """Captioner + vocab + q_vocab ro az QD checkpoint load kon.

    Priority baraye q_vocab / q_emb:
        1) QD captioner ``best.pt`` (key ``q_vocab`` + ``q_emb``/``q_gru``)
        2) optional ``--vqa-ckpt`` (legacy override)

    Finglish: hyperparam haye model az ``state['config']`` (age bashe) roye YAML
    override mishan ta dims/model_class ba checkpoint yeki bashe.

    Returns:
        (model, caption_vocab, q_vocab)
    """
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cap_state = state.get("model", state)
    if "vocab" not in state:
        raise ValueError(f"No vocab in checkpoint: {ckpt_path}")
    vocab = vocab_from_itos(state["vocab"])

    if "q_vocab" in state and isinstance(state["q_vocab"], list):
        q_vocab = vocab_from_itos(state["q_vocab"])
        question_vocab_size = len(q_vocab.itos)
    elif any(k.startswith("q_emb.") for k in cap_state.keys()):
        question_vocab_size = int(cap_state["q_emb.weight"].shape[0])
        q_vocab = Vocab.__new__(Vocab)
        q_vocab.itos = [f"<tok_{i}>" for i in range(question_vocab_size)]
        q_vocab.itos[0], q_vocab.itos[1], q_vocab.itos[2], q_vocab.itos[3] = (
            "<pad>",
            "<bos>",
            "<eos>",
            "<unk>",
        )
        q_vocab.stoi = {w: i for i, w in enumerate(q_vocab.itos)}
    else:
        raise ValueError(
            f"QD checkpoint needs q_vocab or q_emb weights: {ckpt_path}"
        )

    if vqa_ckpt is not None:
        vqa = torch.load(vqa_ckpt, map_location="cpu")
        if "q_vocab" in vqa:
            q_vocab = vocab_from_itos(vqa["q_vocab"])
            question_vocab_size = len(q_vocab.itos)

    model_cfg = {**cfg, **(state.get("config") or {})}
    if isinstance(state.get("config"), dict) and not state["config"].get("model_class"):
        # Finglish: ckpt ghadimi (ghabl az registry) model_class nadare → hamishe v1
        model_cfg["model_class"] = DEFAULT_MODEL_CLASS
    model = build_captioner(model_cfg, vocab, q_vocab)
    missing, unexpected = model.load_state_dict(cap_state, strict=False)
    # Finglish: detector freeze (pretrained) dar ckpt haye v2 save nemishe → missing OK
    missing = [k for k in missing if not k.startswith("region_encoder.detector.")]
    if missing:
        print(
            f"Note: missing keys when loading captioner ({len(missing)}): "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    if unexpected:
        print(
            f"Note: unexpected keys ({len(unexpected)}): "
            f"{unexpected[:5]}{'...' if len(unexpected) > 5 else ''}"
        )

    if vqa_ckpt is not None:
        vqa_model = torch.load(vqa_ckpt, map_location="cpu").get("model", {})
        q_only = {
            k.replace("captioner.", "", 1): v
            for k, v in vqa_model.items()
            if k.startswith("captioner.q_emb.")
            or k.startswith("captioner.q_proj.")
            or k.startswith("captioner.q_gru.")
        }
        if q_only:
            model.load_state_dict(q_only, strict=False)

    model.eval().to(device)
    return model, vocab, q_vocab


def split_paths(cfg: Dict[str, Any], split: str) -> Tuple[str, str, str]:
    """(images_dir, filename_template, captions_json_path) baraye train/val/test.

    Finglish: test hamoon val path ha ro dare (row_range az train JSON joda mikone).
    """
    if split == "train":
        return (
            cfg["train_images_dir"],
            cfg["train_image_filename_template"],
            cfg["train_captions_json"],
        )
    if split in ("val", "test"):
        return (
            cfg["val_images_dir"],
            cfg["val_image_filename_template"],
            cfg["val_captions_json"],
        )
    raise ValueError(f"split must be train, val or test, got {split!r}")


def qd_rows_for_image(qd_json: str, image_id: int) -> List[Dict[str, Any]]:
    """Hame QD sample haye yek image_id (question + caption)."""
    return [r for r in load_qd_json(qd_json) if int(r["image_id"]) == int(image_id)]


def run_single(
    model: nn.Module,
    vocab: Vocab,
    cfg: Dict[str, Any],
    device: torch.device,
    image_id: int,
    split: str,
    question: Optional[str],
    q_vocab: Vocab,
) -> None:
    """Yek sample: ``image_id`` + soal → caption + GT chap kon."""
    images_dir, template, captions_json = split_paths(cfg, split)
    img_path = image_path(image_id, images_dir, template)
    image = load_image(
        image_id, images_dir, template, device, image_size_from_cfg(cfg)
    )

    if not question:
        rows = qd_rows_for_image(captions_json, image_id)
        if not rows:
            raise ValueError(
                f"No QD rows for image_id={image_id} in {captions_json}"
            )
        question = str(rows[0]["question"])
        print(f"(auto) using first QD question for image_id={image_id}")

    max_q = int(cfg.get("max_question_len", cfg.get("max_caption_len", 14)))
    q_ids = encode_question(question, q_vocab, max_q, device)

    with torch.no_grad():
        cache_dir = region_cache_dir_for_split(cfg, split)
        img_ids = torch.tensor([image_id], dtype=torch.long, device=device)
        cap_ids = model.generate_caption(
            image,
            q_ids,
            int(cfg["max_caption_len"]),
            image_ids=img_ids,
            region_cache_dir=cache_dir,
            save_region_cache=True,
        )
    pred = decode_ids(cap_ids[0].tolist(), vocab)

    print(f"split={split}  dataset=qd")
    print(f"image_id: {image_id}")
    print(f"image_file: {img_path.name}")
    print(f"question: {question}")
    print(f"pred: {pred}")

    rows = qd_rows_for_image(captions_json, image_id)
    match = [
        r
        for r in rows
        if str(r["question"]).strip().lower() == question.strip().lower()
    ]
    gt = str(match[0]["caption"]) if match else None
    print(f"gt caption: {gt}")


def run_val_qd(
    model: nn.Module,
    vocab: Vocab,
    q_vocab: Vocab,
    cfg: Dict[str, Any],
    device: torch.device,
    n_samples: int,
    split: str,
    metric_images: int,
) -> None:
    """Eval QD: teacher-forcing loss/token_acc + BLEU/CIDEr question-guided."""
    images_dir, filename_template, _ = split_paths(cfg, split)
    # Finglish: hamoon split builder train.py — ``<split>_row_range`` (image-disjoint,
    # bedoon leak) age set bashe; vagarna max_*_images / max_*_samples (ghadimi).
    ds = build_split_dataset(cfg, split, vocab, q_vocab)
    loader = DataLoader(
        ds,
        batch_size=int(cfg.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
        collate_fn=collate_batch,
    )

    loss, acc = eval_epoch(
        model, loader, nn.CrossEntropyLoss(ignore_index=0), device, cfg, split=split
    )
    print(
        f"{split}_loss (teacher forcing, QD): {loss:.4f}  "
        f"{split}_token_acc: {acc:.4f}  (n_samples={len(ds)})"
    )

    rows = ds.samples
    if metric_images > 0:
        rows = rows[:metric_images]

    hyps: List[List[str]] = []
    refs: List[List[List[str]]] = []
    sample_out: List[Dict[str, Any]] = []
    cache_dir = region_cache_dir_for_split(cfg, split)
    max_q = int(cfg.get("max_question_len", 14))

    for s in rows:
        image_id = int(s["image_id"])
        question = str(s["question"])
        gt_cap = str(s["caption"])
        image = load_image(
            image_id, images_dir, filename_template, device, image_size_from_cfg(cfg)
        )
        q_ids = encode_question(question, q_vocab, max_q, device)
        with torch.no_grad():
            img_ids = torch.tensor([image_id], dtype=torch.long, device=device)
            cap_ids = model.generate_caption(
                image,
                q_ids,
                int(cfg["max_caption_len"]),
                image_ids=img_ids,
                region_cache_dir=cache_dir,
                save_region_cache=True,
            )
        pred = decode_ids(cap_ids[0].tolist(), vocab)
        hyps.append(tok(pred))
        refs.append([tok(gt_cap)])
        sample_out.append(
            {
                "image_id": image_id,
                "question": question,
                "pred": pred,
                "gt": gt_cap,
            }
        )

    if hyps:
        scores = compute_caption_metrics(hyps, refs)
        score_str = "  ".join(f"{k}={v:.4f}" for k, v in scores.items())
        print(
            f"\nGenerated QD-caption metrics "
            f"(n_pairs={len(hyps)}, question-guided):\n  {score_str}"
        )

    if n_samples <= 0:
        return

    print(f"\nSample QD captions (split={split}, n={min(n_samples, len(sample_out))}):")
    for row in sample_out[:n_samples]:
        img_path = image_path(row["image_id"], images_dir, filename_template)
        print(f"\nimage_id: {row['image_id']}")
        print(f"image_file: {img_path.name}")
        print(f"question: {row['question']}")
        print(f"pred: {row['pred']}")
        print(f"gt:   {row['gt']}")


def parse_args() -> argparse.Namespace:
    """Argument haye CLI ro parse kon."""
    p = argparse.ArgumentParser(
        description="QD captioner — eval / infer"
    )
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--ckpt", required=True, help="Captioner checkpoint (.pt)")
    p.add_argument(
        "--vqa-ckpt",
        default=None,
        help="Optional VQA best.pt — override/load captioner.q_* for --question",
    )
    p.add_argument("--image-id", type=int, default=None, help="Single COCO image_id")
    p.add_argument("--split", choices=("train", "val", "test"), default="val")
    p.add_argument("--question", default=None, help="Question text (required for QD)")
    p.add_argument(
        "--samples",
        type=int,
        default=0,
        help="Without --image-id: show N caption examples (default 10 if 0)",
    )
    p.add_argument(
        "--metric-images",
        type=int,
        default=500,
        help="Max pairs for BLEU/CIDEr (0 = all available in loader cap)",
    )
    return p.parse_args()


def main() -> None:
    """Entry: config → model → single sample ya split metrics."""
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    cfg = load_config(str(config_path))
    resolve_path_fields(cfg, PATH_KEYS)
    set_seed(int(cfg.get("seed", 42)))

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.get("device") == "cuda" else "cpu"
    )
    ckpt_path = Path(args.ckpt).expanduser().resolve()
    vqa_ckpt = Path(args.vqa_ckpt).expanduser().resolve() if args.vqa_ckpt else None

    model, vocab, q_vocab = build_model_from_ckpt(ckpt_path, cfg, device, vqa_ckpt)
    print(
        f"config={config_path}  ckpt={ckpt_path}  device={device}  "
        f"dataset=qd  vocab={len(vocab.itos)} q_vocab={len(q_vocab.itos)}"
    )

    if args.image_id is not None:
        run_single(
            model,
            vocab,
            cfg,
            device,
            args.image_id,
            args.split,
            args.question,
            q_vocab,
        )
    else:
        n = args.samples if args.samples > 0 else 10
        run_val_qd(
            model,
            vocab,
            q_vocab,
            cfg,
            device,
            n,
            split=args.split,
            metric_images=args.metric_images,
        )


if __name__ == "__main__":
    main()
