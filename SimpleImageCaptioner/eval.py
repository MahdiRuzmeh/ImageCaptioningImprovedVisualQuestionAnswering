"""SimpleImageCaptioner — eval va inference (marhale 1 paper).

In file joda az ``train.py`` hast. Ba ``best.pt`` mitooni:
- yek ``image_id`` bedi → **caption** begiri
- (optional) soal bedi → caption **question-guided** (faghat ba ``q_emb`` az VQA ckpt)
- roye val split **loss** + chand sample caption (pred vs ground truth) bebini

Chera in file?
--------------
``train.py`` faghat train/eval kol rooye dataset anjam mide.
``eval.py`` baraye **test dasti** hast: yek aks, yek soal, ya chand sample.

Input / Output
--------------
- **Input:** ``image_id`` (COCO), optional ``--question``, optional ``--split`` (train/val)
- **Output:** caption (greedy decode) + ground truth agar toye JSON bashe

Pish-niaz
---------
- Checkpoint captioner: ``outputs/<run>/best.pt`` (bayad key ``vocab`` dashte bashe)
- Config hamooni ke train kardi (path dataset, ``max_caption_len``, ...)
- Run az folder ``SimpleImageCaptioner/``
- Python env ba ``torch`` + ``torchvision`` (mesl ``src/.venv``)

CLI arguments
-------------
``--config``       path be YAML (default: ``configs/default.yaml``)
``--ckpt``         **required** — captioner ``best.pt``
``--image-id``     yek COCO image_id (age bedi → mode single sample)
``--split``        ``train`` ya ``val`` (default: ``val``)
``--question``     optional — matn soal baraye question-guided caption
``--vqa-ckpt``     optional — VQA ``best.pt`` baraye load ``q_emb``/``q_proj``
``--samples``      ba val mode (bedoon ``--image-id``): tedad sample caption (default 10)

Che mode ee ejra mishe?
-----------------------
1. **Single sample** — ``--image-id`` set shode
2. **Val metrics** — ``--image-id`` nist → loss roye val + ``--samples`` ta mesal

Nokte: baraye ``--question`` hatman ``--vqa-ckpt`` bede (``q_emb`` to marhale 2 train shode).

Chand mesal (local smoke)
-------------------------
::

    cd SimpleImageCaptioner

    # 1) yek aks → caption
    python eval.py --config configs/smoke.yaml --ckpt outputs/smoke/best.pt \\
        --image-id 203564 --split val

    # 2) soal + caption (question-guided)
    python eval.py --config configs/smoke.yaml --ckpt outputs/smoke/best.pt \\
        --image-id 262148 --split val --question "Where is he looking?" \\
        --vqa-ckpt ../SimpleVQA/outputs/smoke/best.pt

    # 3) val loss + 10 sample caption
    python eval.py --config configs/smoke.yaml --ckpt outputs/smoke/best.pt \\
        --split val --samples 10

Chand mesal (Kaggle mini)
-------------------------
::

    python eval.py --config configs/kaggle_mini.yaml --ckpt outputs/kaggle_mini/best.pt \\
        --image-id 391895 --split val

    python eval.py --config configs/kaggle_mini.yaml --ckpt outputs/kaggle_mini/best.pt \\
        --image-id 391895 --question "what color is the bus?" \\
        --vqa-ckpt ../SimpleVQA/outputs/kaggle_mini/best.pt

    python eval.py --config configs/kaggle_mini.yaml --ckpt outputs/kaggle_mini/best.pt \\
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

from models.captioner_v1 import SimpleImageCaptioner
from train import (
    CocoCaptionDataset,
    PROJECT_ROOT,
    Vocab,
    collate_batch,
    eval_epoch,
    image_cap,
    load_caps_json,
    load_config,
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
    """``Vocab`` ro az list ``itos`` toye checkpoint dobare besaz.

    Chera lazeme?
        - train vocab ro toye ``best.pt`` save mikone
        - eval bayad **hamoon** index-ha ro estefade kone (na rebuild az JSON)

    Input:
        itos: list kalamat/vocab ke toye checkpoint save shode

    Output:
        Vocab object ba ``itos`` va ``stoi``
    """
    v = Vocab.__new__(Vocab)
    v.itos = list(itos)
    v.stoi = {w: i for i, w in enumerate(v.itos)}
    return v


def decode_ids(ids: List[int], vocab: Vocab) -> str:
    """Token id ha ro be matn tabdil kon.

    PAD (0), BOS (1), EOS (2) va token haye khar ro skip mikonim.
    Baraye chap caption ya debug estefade mishe.
    """
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
    """Yek COCO image ro load kon va preprocess kon.

    Input:
        image_id: mesl 203564
        images_dir: folder val2014/train2014
        filename_template: mesl ``COCO_val2014_{image_id:012d}.jpg``
        device: cuda ya cpu
        image_size: az config (``image_size``) — hamoon abaad train

    Output:
        tensor ba shape (1, 3, image_size, image_size)
    """
    path = Path(images_dir) / filename_template.format(image_id=image_id)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    tensor = image_transform(image_size)(Image.open(path).convert("RGB"))
    return tensor.unsqueeze(0).to(device)


def encode_question(
    question: str, q_vocab: Vocab, max_len: int, device: torch.device
) -> torch.Tensor:
    """Matn soal ro encode kon baraye ``q_emb``.

    Format: BOS (id=1) + token haye soal + EOS (id=2)
    Hamoon convention VQA/train.

    Output:
        tensor (1, seq_len) roye device
    """
    ids = [1] + q_vocab.encode(tok(question)[: max_len - 2]) + [2]
    return torch.tensor([ids], dtype=torch.long, device=device)


def build_model_from_ckpt(
    ckpt_path: Path,
    cfg: Dict[str, Any],
    device: torch.device,
    vqa_ckpt: Optional[Path] = None,
) -> Tuple[SimpleImageCaptioner, Vocab, Optional[Vocab]]:
    """Captioner + vocab ro az checkpoint load kon.

    Marhale 1 (faghat captioner ckpt):
        - ``word_emb``, LSTM, attention, ... az ``best.pt`` captioner

    Marhale 2 (ba ``vqa_ckpt``):
        - ``q_vocab`` az VQA checkpoint
        - ``q_emb`` va ``q_proj`` az ``captioner.q_emb.*`` toye VQA ckpt
        - in layer ha to VQA train shodan → baraye question-guided caption

    Input:
        ckpt_path: captioner ``best.pt``
        cfg: hyperparams az YAML
        vqa_ckpt: optional VQA ``best.pt``

    Output:
        (model, caption_vocab, q_vocab ya None)
    """
    state = torch.load(ckpt_path, map_location="cpu")
    cap_state = state.get("model", state)
    vocab = vocab_from_itos(state["vocab"]) if "vocab" in state else None
    if vocab is None:
        raise ValueError(f"No vocab in checkpoint: {ckpt_path}")

    q_vocab: Optional[Vocab] = None
    question_vocab_size: Optional[int] = None
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
    )
    model.load_state_dict(cap_state, strict=False)

    if vqa_ckpt is not None and question_vocab_size is not None:
        vqa_model = torch.load(vqa_ckpt, map_location="cpu").get("model", {})
        q_only = {
            k.replace("captioner.", "", 1): v
            for k, v in vqa_model.items()
            if k.startswith("captioner.q_emb.") or k.startswith("captioner.q_proj.")
        }
        model.load_state_dict(q_only, strict=False)

    model.eval().to(device)
    return model, vocab, q_vocab


def split_paths(cfg: Dict[str, Any], split: str) -> Tuple[str, str, str]:
    """Az config, path image dir + template + captions JSON ro baraye train/val bargardoon.

    Input:
        split: ``"train"`` ya ``"val"``

    Output:
        (images_dir, filename_template, captions_json_path)
    """
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
    """Yek sample: ``image_id`` (+ optional soal) → caption ro chap kon.

    Flow:
        1. image load
        2. agar soal dasht → encode soal (age q_emb load shode)
        3. ``generate_caption`` (greedy)
        4. chap pred + ground truth (age toye JSON bashe)

    Agar ``--question`` bedi vali ``q_emb`` load nashode → warning chap mishe
    va caption bedoon soal tolid mishe.
    """
    images_dir, template, captions_json = split_paths(cfg, split)
    image = load_image(
        image_id, images_dir, template, device, image_size_from_cfg(cfg)
    )

    q_ids: Optional[torch.Tensor] = None
    if question:
        if q_vocab is None or model.q_emb is None:
            print(
                "Warning: --question given but no q_emb loaded. "
                "Pass --vqa-ckpt (VQA best.pt) for question-guided captions."
            )
        else:
            q_ids = encode_question(
                question, q_vocab, int(cfg["max_caption_len"]), device
            )

    with torch.no_grad():
        cap_ids = model.generate_caption(image, q_ids, int(cfg["max_caption_len"]))
    pred = decode_ids(cap_ids[0].tolist(), vocab)

    gt_caps = load_caps_json(captions_json).get(image_id, [])

    print(f"image_id={image_id}  split={split}")
    if question:
        print(f"question: {question}")
    print(f"caption: {pred}")
    if gt_caps:
        print(f"ground_truth (first): {gt_caps[0]}")


def run_val(
    model: SimpleImageCaptioner,
    vocab: Vocab,
    cfg: Dict[str, Any],
    device: torch.device,
    n_samples: int,
    split: str = "val",
) -> None:
    """Roye train ya val split: teacher-forcing loss + N ta greedy caption sample.

    Ghabl az sample ha:
        - loss ba teacher forcing (mesl ``train.py`` eval_epoch)

    Bad:
        - N ta image unique → greedy caption vs ground truth chap mishe

    ``max_train_images`` / ``max_val_images`` az config cap mikone (mesl kaggle_mini/smoke).
    """
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
        model, loader, nn.CrossEntropyLoss(ignore_index=0), device, cfg
    )
    print(
        f"{split}_loss (teacher forcing): {loss:.4f}  {split}_token_acc: {acc:.4f}"
    )

    if n_samples <= 0:
        return

    seen: set[int] = set()
    examples: List[Tuple[int, str]] = []
    for image_id, caption in ds.samples:
        if image_id in seen:
            continue
        seen.add(image_id)
        examples.append((image_id, caption))
        if len(examples) >= n_samples:
            break

    print(f"\nSample greedy captions (split={split}, n={len(examples)}):")
    for image_id, gt in examples:
        image = load_image(
            image_id,
            images_dir,
            filename_template,
            device,
            image_size_from_cfg(cfg),
        )
        with torch.no_grad():
            cap_ids = model.generate_caption(
                image, None, int(cfg["max_caption_len"])
            )
        pred = decode_ids(cap_ids[0].tolist(), vocab)
        print(f"  [{image_id}] pred: {pred}")
        print(f"           gt:   {gt}")


def parse_args() -> argparse.Namespace:
    """Argument haye CLI ro parse kon.

    Bebin module docstring bala baraye jadval kamel argument-ha va mesal-ha.
    """
    p = argparse.ArgumentParser(description="SimpleImageCaptioner — eval / infer")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--ckpt", required=True, help="Captioner checkpoint (.pt)")
    p.add_argument(
        "--vqa-ckpt",
        default=None,
        help="VQA best.pt — load captioner.q_emb for --question",
    )
    p.add_argument("--image-id", type=int, default=None, help="Single COCO image_id")
    p.add_argument("--split", choices=("train", "val"), default="val")
    p.add_argument("--question", default=None, help="Optional question text")
    p.add_argument(
        "--samples",
        type=int,
        default=0,
        help="If set without --image-id, show N greedy caption examples on --split",
    )
    return p.parse_args()


def main() -> None:
    """Entry point: config load → model load → single sample ya val metrics.

    Age ``--image-id`` bashe → ``run_single``
    Age nabashe → ``run_val`` (loss + samples)
    """
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
    print(f"config={config_path}  ckpt={ckpt_path}  device={device}")

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
        run_val(model, vocab, cfg, device, n, split=args.split)


if __name__ == "__main__":
    main()
