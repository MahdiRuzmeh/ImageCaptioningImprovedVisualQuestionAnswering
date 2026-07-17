"""SimpleImageCaptioner — eval va inference (marhale 1 paper + QD).

Ba ``best.pt`` mitooni:
- yek ``image_id`` (+ optional soal) → caption
- roye split: **token accuracy** (teacher forcing) + BLEU/CIDEr + sample ha
- QD checkpoint: ``q_vocab`` + ``q_emb``/``q_gru`` az hamoon ``best.pt`` (bedoon VQA ckpt)

Run az ``SimpleImageCaptioner/``::

    # QD smoke — accuracy + metrics + samples
    python eval.py --config configs/smoke.yaml --ckpt outputs/smoke/best.pt \\
        --split val --samples 10

    # QD — yek sample ba soal
    python eval.py --config configs/smoke.yaml --ckpt outputs/smoke/best.pt \\
        --image-id 9 --split train --question "How many cookies can be seen?"

    # Legacy MSCOCO
    python eval.py --config configs/smoke.yaml --ckpt outputs/smoke/best.pt \\
        --split val --samples 10
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
from models.captioner_v1 import SimpleImageCaptioner
from train import (
    CocoCaptionDataset,
    PROJECT_ROOT,
    Vocab,
    VqaQdCaptionDataset,
    collate_batch,
    eval_epoch,
    image_cap,
    load_caps_json,
    load_config,
    load_qd_json,
    region_cache_dir_for_split,
    resolve_path_fields,
    set_seed,
    tok,
)

# path haye config ke bayad absolute beshan (relative be cwd)
PATH_KEYS = (
    "train_captions_json",
    "val_captions_json",
    "train_images_dir",
    "val_images_dir",
    "save_dir",
    "train_region_cache_dir",
    "val_region_cache_dir",
)


def image_size_from_cfg(cfg: Dict[str, Any]) -> int:
    """Finglish: image_size ro az YAML migirim (default 448) ta ba train yeksan bashe."""
    return int(cfg.get("image_size", 448))


def is_qd_mode(cfg: Dict[str, Any]) -> bool:
    """True when training data is question-dependent caption JSON."""
    if "question_dependent" in str(cfg.get("train_captions_json", "")):
        return True
    return str(cfg.get("dataset_mode", "qd")).lower() == "qd"


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
    """``Vocab`` ro az list ``itos`` toye checkpoint dobare besaz.

    Chera lazeme?
        - train vocab ro toye ``best.pt`` save mikone
        - eval bayad **hamoon** index-ha ro estefade kone (na rebuild az JSON)
    """
    v = Vocab.__new__(Vocab)
    v.itos = list(itos)
    v.stoi = {w: i for i, w in enumerate(v.itos)}
    return v


def decode_ids(ids: List[int], vocab: Vocab) -> str:
    """Token id ha ro be matn tabdil kon (PAD/BOS/EOS skip)."""
    words = [
        vocab.itos[i]
        for i in ids
        if 0 < i < len(vocab.itos) and vocab.itos[i] not in (vocab.PAD, vocab.BOS, vocab.EOS)
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
    """Absolute-ish image path for a COCO image_id (based on config template)."""
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
) -> Tuple[SimpleImageCaptioner, Vocab, Optional[Vocab]]:
    """Captioner + vocab (+ optional q_vocab) ro az checkpoint load kon.

    Priority baraye q_vocab / q_emb:
        1) QD captioner ``best.pt`` (key ``q_vocab`` + ``q_emb``/``q_gru`` dar state)
        2) optional ``--vqa-ckpt`` (legacy: load q layers az VQA)

    Output:
        (model, caption_vocab, q_vocab ya None)
    """
    state = torch.load(ckpt_path, map_location="cpu")
    cap_state = state.get("model", state)
    if "vocab" not in state:
        raise ValueError(f"No vocab in checkpoint: {ckpt_path}")
    vocab = vocab_from_itos(state["vocab"])

    q_vocab: Optional[Vocab] = None
    question_vocab_size: Optional[int] = None

    # QD Stage-1: q_vocab toye hamoon captioner ckpt save shode
    if "q_vocab" in state and isinstance(state["q_vocab"], list):
        q_vocab = vocab_from_itos(state["q_vocab"])
        question_vocab_size = len(q_vocab.itos)
    elif any(k.startswith("q_emb.") for k in cap_state.keys()):
        # weight hast vali itos nist → size az weight
        question_vocab_size = int(cap_state["q_emb.weight"].shape[0])

    if vqa_ckpt is not None:
        vqa = torch.load(vqa_ckpt, map_location="cpu")
        if "q_vocab" in vqa:
            q_vocab = vocab_from_itos(vqa["q_vocab"])
            question_vocab_size = len(q_vocab.itos)

    hidden_dim = int(cfg.get("hidden_dim", cfg.get("lstm_hidden", 512)))
    model = SimpleImageCaptioner(
        vocab_size=len(vocab.itos),
        pad_id=vocab.pad_id,
        word_dim=int(cfg["word_dim"]),
        hidden_dim=hidden_dim,
        max_regions=int(cfg["max_regions"]),
        question_dim=int(cfg.get("question_dim", cfg["word_dim"])),
        embed_dim=int(cfg.get("embed_dim", hidden_dim)),
        region_dim=int(cfg.get("region_dim", 2048)),
        question_vocab_size=question_vocab_size,
        question_pad_id=q_vocab.pad_id if q_vocab else vocab.pad_id,
        dropout=float(cfg.get("dropout", 0.5)),
        use_gnn=bool(cfg.get("use_gnn", True)),
        gnn_dim=int(cfg["gnn_dim"]) if cfg.get("gnn_dim") is not None else None,
    )
    # strict=False: old coco ckpt / partial keys
    missing, unexpected = model.load_state_dict(cap_state, strict=False)
    if missing:
        print(f"Note: missing keys when loading captioner ({len(missing)}): "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"Note: unexpected keys ({len(unexpected)}): "
              f"{unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    if vqa_ckpt is not None and question_vocab_size is not None:
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
    """(images_dir, filename_template, captions_json_path) baraye train/val."""
    if split == "train":
        return (
            cfg["train_images_dir"],
            cfg["train_image_filename_template"],
            cfg["train_captions_json"],
        )
    if split == "val":
        return (
            cfg["val_images_dir"],
            cfg["val_image_filename_template"],
            cfg["val_captions_json"],
        )
    raise ValueError(f"split must be train or val, got {split!r}")


def qd_rows_for_image(qd_json: str, image_id: int) -> List[Dict[str, Any]]:
    """Hame QD sample haye yek image_id (question + caption)."""
    return [r for r in load_qd_json(qd_json) if int(r["image_id"]) == int(image_id)]


def run_single(
    model: SimpleImageCaptioner,
    vocab: Vocab,
    cfg: Dict[str, Any],
    device: torch.device,
    image_id: int,
    split: str,
    question: Optional[str],
    q_vocab: Optional[Vocab],
) -> None:
    """Yek sample: ``image_id`` (+ optional soal) → caption + GT chap kon."""
    images_dir, template, captions_json = split_paths(cfg, split)
    img_path = image_path(image_id, images_dir, template)
    image = load_image(
        image_id, images_dir, template, device, image_size_from_cfg(cfg)
    )

    # QD: age soal nadashtim, avalin soal hamoon image ro az JSON bardar
    qd = is_qd_mode(cfg)
    if qd and not question:
        rows = qd_rows_for_image(captions_json, image_id)
        if rows:
            question = str(rows[0]["question"])
            print(f"(auto) using first QD question for image_id={image_id}")

    q_ids: Optional[torch.Tensor] = None
    max_q = int(cfg.get("max_question_len", cfg.get("max_caption_len", 14)))
    if question:
        if q_vocab is None or model.q_emb is None:
            print(
                "Warning: --question given but no q_emb/q_vocab loaded. "
                "Use a QD captioner ckpt (with q_vocab) or pass --vqa-ckpt."
            )
        else:
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

    print(f"split={split}  dataset_mode={'qd' if qd else 'coco'}")
    print(f"image_id: {image_id}")
    print(f"image_file: {img_path.name}")
    if question:
        print(f"question: {question}")
    print(f"pred: {pred}")

    if qd:
        rows = qd_rows_for_image(captions_json, image_id)
        if question:
            match = [r for r in rows if str(r["question"]).strip().lower()
                     == question.strip().lower()]
            gt = str(match[0]["caption"]) if match else None
            print(f"gt caption: {gt}")
        else:
            for i, r in enumerate(rows[:5]):
                print(f"gt[{i}] Q: {r['question']}  C: {r['caption']}")
    else:
        gt_caps = load_caps_json(captions_json).get(image_id, [])
        print(f"gt caption[0]: {gt_caps[0] if gt_caps else None}")
        print(f"gt caption[4]: {gt_caps[4] if len(gt_caps) > 4 else None}")


def run_val_coco(
    model: SimpleImageCaptioner,
    vocab: Vocab,
    cfg: Dict[str, Any],
    device: torch.device,
    n_samples: int,
    split: str,
    metric_images: int,
) -> None:
    """Eval MSCOCO: teacher-forcing loss/acc + BLEU/CIDEr (bedoon soal)."""
    images_dir, filename_template, captions_json = split_paths(cfg, split)
    cap_key = "max_train_images" if split == "train" else "max_val_images"
    max_images = image_cap(cfg.get(cap_key))
    image_ids = None
    if max_images is not None:
        image_ids = sorted(load_caps_json(captions_json).keys())[:max_images]

    ds = CocoCaptionDataset(
        images_dir,
        captions_json,
        vocab,
        int(cfg["max_caption_len"]),
        filename_template,
        image_ids=image_ids,
        image_size=image_size_from_cfg(cfg),
    )
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
        f"{split}_loss (teacher forcing): {loss:.4f}  {split}_token_acc: {acc:.4f}"
    )

    caps_by_img = load_caps_json(captions_json)
    unique_ids: List[int] = []
    seen_m: set[int] = set()
    for image_id, _ in ds.samples:
        if image_id in seen_m:
            continue
        seen_m.add(image_id)
        unique_ids.append(image_id)
        if metric_images > 0 and len(unique_ids) >= metric_images:
            break

    hyps: List[List[str]] = []
    refs: List[List[List[str]]] = []
    pred_by_id: Dict[int, str] = {}
    cache_dir = region_cache_dir_for_split(cfg, split)
    for image_id in unique_ids:
        image = load_image(
            image_id, images_dir, filename_template, device, image_size_from_cfg(cfg)
        )
        with torch.no_grad():
            img_ids = torch.tensor([image_id], dtype=torch.long, device=device)
            cap_ids = model.generate_caption(
                image,
                None,
                int(cfg["max_caption_len"]),
                image_ids=img_ids,
                region_cache_dir=cache_dir,
                save_region_cache=True,
            )
        pred = decode_ids(cap_ids[0].tolist(), vocab)
        pred_by_id[image_id] = pred
        hyps.append(tok(pred))
        refs.append([tok(c) for c in caps_by_img.get(image_id, [])])

    if hyps:
        scores = compute_caption_metrics(hyps, refs)
        score_str = "  ".join(f"{k}={v:.4f}" for k, v in scores.items())
        print(f"\nGenerated-caption metrics (n_images={len(hyps)}):\n  {score_str}")

    if n_samples <= 0:
        return

    examples: List[Tuple[int, str]] = []
    seen: set[int] = set()
    for image_id, caption in ds.samples:
        if image_id in seen:
            continue
        seen.add(image_id)
        examples.append((image_id, caption))
        if len(examples) >= n_samples:
            break

    print(f"\nSample greedy captions (split={split}, n={len(examples)}):")
    for image_id, _gt in examples:
        img_path = image_path(image_id, images_dir, filename_template)
        pred = pred_by_id.get(image_id, "")
        all_gt = caps_by_img.get(image_id, [])
        print(f"\nimage_id: {image_id}")
        print(f"image_file: {img_path.name}")
        print(f"pred: {pred}")
        print(f"gt caption[0]: {all_gt[0] if all_gt else None}")
        print(f"gt caption[4]: {all_gt[4] if len(all_gt) > 4 else None}")


def run_val_qd(
    model: SimpleImageCaptioner,
    vocab: Vocab,
    q_vocab: Vocab,
    cfg: Dict[str, Any],
    device: torch.device,
    n_samples: int,
    split: str,
    metric_images: int,
) -> None:
    """Eval QD: teacher-forcing loss/token_acc ba soal + BLEU/CIDEr question-guided.

    Har sample = (image, question, caption). Generate ba ``question_ids`` anjam mishe.
    """
    if model.q_emb is None or model.q_gru is None:
        raise RuntimeError(
            "QD eval needs q_emb/q_gru. Load a QD captioner checkpoint "
            "(trained with dataset_mode: qd)."
        )

    images_dir, filename_template, qd_json = split_paths(cfg, split)
    cap_key = "max_train_images" if split == "train" else "max_val_images"
    samp_key = "max_train_samples" if split == "train" else "max_val_samples"
    max_images = image_cap(cfg.get(cap_key))
    max_samples = image_cap(cfg.get(samp_key))

    image_ids = None
    if max_images is not None:
        image_ids = sorted(
            {int(r["image_id"]) for r in load_qd_json(qd_json)}
        )[:max_images]

    ds = VqaQdCaptionDataset(
        images_dir,
        qd_json,
        vocab,
        q_vocab,
        int(cfg["max_caption_len"]),
        int(cfg.get("max_question_len", 14)),
        filename_template,
        image_ids=image_ids,
        image_size=image_size_from_cfg(cfg),
        max_samples=max_samples,
    )
    loader = DataLoader(
        ds,
        batch_size=int(cfg.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
        collate_fn=collate_batch,
    )

    # teacher forcing loss + token accuracy (ba question_ids)
    loss, acc = eval_epoch(
        model, loader, nn.CrossEntropyLoss(ignore_index=0), device, cfg, split=split
    )
    print(
        f"{split}_loss (teacher forcing, QD): {loss:.4f}  "
        f"{split}_token_acc: {acc:.4f}  (n_samples={len(ds)})"
    )

    # generated metrics: ta metric_images ta (image, question) pair
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


def run_val(
    model: SimpleImageCaptioner,
    vocab: Vocab,
    cfg: Dict[str, Any],
    device: torch.device,
    n_samples: int,
    split: str = "val",
    metric_images: int = 500,
    q_vocab: Optional[Vocab] = None,
) -> None:
    """Dispatch: QD ya MSCOCO eval baraye accuracy + BLEU/CIDEr."""
    if is_qd_mode(cfg):
        if q_vocab is None:
            raise RuntimeError(
                "QD config needs q_vocab in captioner checkpoint. "
                "Train with configs/qd_*.yaml first."
            )
        run_val_qd(
            model, vocab, q_vocab, cfg, device, n_samples, split, metric_images
        )
    else:
        run_val_coco(
            model, vocab, cfg, device, n_samples, split, metric_images
        )


def parse_args() -> argparse.Namespace:
    """Argument haye CLI ro parse kon."""
    p = argparse.ArgumentParser(description="SimpleImageCaptioner — eval / infer")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--ckpt", required=True, help="Captioner checkpoint (.pt)")
    p.add_argument(
        "--vqa-ckpt",
        default=None,
        help="Optional VQA best.pt — override/load captioner.q_* for --question",
    )
    p.add_argument("--image-id", type=int, default=None, help="Single COCO image_id")
    p.add_argument("--split", choices=("train", "val"), default="val")
    p.add_argument("--question", default=None, help="Optional question text (QD/guided)")
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
        help="Max images/pairs for BLEU/CIDEr (0 = all available in loader cap)",
    )
    return p.parse_args()


def main() -> None:
    """Entry: config → model → single sample ya split metrics (accuracy)."""
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
    mode = "qd" if is_qd_mode(cfg) else "coco"
    q_info = f" q_vocab={len(q_vocab.itos)}" if q_vocab is not None else ""
    print(
        f"config={config_path}  ckpt={ckpt_path}  device={device}  "
        f"dataset_mode={mode}  vocab={len(vocab.itos)}{q_info}"
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
        run_val(
            model,
            vocab,
            cfg,
            device,
            n,
            split=args.split,
            metric_images=args.metric_images,
            q_vocab=q_vocab,
        )


if __name__ == "__main__":
    main()
