"""Minimal caption checkpoint smoke test after stage-one training.

Purpose
-------
Not a full MSCOCO caption benchmark—loads ``best.pt`` / ``last.pt``, runs greedy decoding on
validation samples, prints captions, and optionally saves/shows the image beside them.
Use BLEU/CIDEr notebooks for thesis-quality metrics.

CLI Examples
------------
::

    cd ImageCaptioner
    python evaluation/evaluate.py --config configs/smoke.yaml --ckpt outputs/smoke/best.pt
    python evaluation/evaluate.py --config configs/smoke.yaml --ckpt outputs/smoke/best.pt \\
        --num-samples 3 --save-dir outputs/smoke/previews --show
"""

import argparse
from pathlib import Path
import sys
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import torch

from datasets.coco_caption_dataset import (
    CocoCaptionDataset,
    Vocab,
    build_vocab,
    load_caps,
    select_image_ids,
    vocab_from_itos,
)
from models.captioner_v1 import ImageCaptionerV1
from utils.common import load_config, resolve_path_fields

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def decode_caption(itos: list, token_ids: list) -> str:
    """Join tokens, skipping pad (0), bos (1), eos (2)."""
    return " ".join(itos[i] for i in token_ids if i > 2)


def denormalize_image(image_chw: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization for display (CHW float tensor in [0, 1])."""
    mean = torch.tensor(IMAGENET_MEAN, device=image_chw.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=image_chw.device).view(3, 1, 1)
    return (image_chw * std + mean).clamp(0.0, 1.0)


def preview_caption_samples(
    model: ImageCaptionerV1,
    dataset: CocoCaptionDataset,
    vocab: Vocab,
    caps_by_image: dict,
    device: torch.device,
    max_caption_len: int,
    num_samples: int = 1,
    save_dir: Optional[Path] = None,
    show: bool = False,
) -> None:
    """Run greedy captioning on validation images and show results next to the image.

    For each of ``num_samples`` **unique** images (in dataset order):

    1. Load one preprocessed tensor from ``dataset`` (448×448, ImageNet-normalized).
    2. Run ``model.generate_caption`` under ``torch.no_grad()`` — autoregressive greedy
       decoding from BOS; no teacher forcing, no gradient updates.
    3. Decode generated token ids with ``vocab.itos``.
    4. Look up **all** MSCOCO reference captions for that ``image_id`` from ``caps_by_image``
       (typically five sentences per COCO image).
    5. Print generated vs reference text to the terminal.
    6. Optionally save a figure (image + caption text) and/or open a matplotlib window.

    Args:
        model: Loaded ``ImageCaptionerV1`` in eval mode.
        dataset: Validation ``CocoCaptionDataset`` (same vocab as the checkpoint).
        vocab: Vocabulary used to decode token ids (from checkpoint when available).
        caps_by_image: ``image_id -> [reference caption strings]`` from ``load_caps``.
        device: CUDA or CPU.
        max_caption_len: Max decode steps passed to ``generate_caption``.
        num_samples: Number of distinct images to preview.
        save_dir: If set, write ``preview_{image_id}.png`` for each sample.
        show: If True, call ``plt.show()`` (blocks until windows are closed).
    """
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    seen_ids: set = set()
    shown = 0

    for idx in range(len(dataset)):
        image_id, _ = dataset.samples[idx]
        if image_id in seen_ids:
            continue
        seen_ids.add(image_id)

        item = dataset[idx]
        images = item["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model.generate_caption(images, max_len=max_caption_len)

        gen_ids = pred[0].tolist()
        generated = decode_caption(vocab.itos, gen_ids)
        references = caps_by_image.get(image_id, [])
        image_name = dataset.image_filename_template.format(image_id=image_id)

        print(f"\n--- Sample {shown + 1} | image_id={image_id} | {image_name} ---")
        print("Generated:", generated)
        print("References:")
        for i, ref in enumerate(references, start=1):
            print(f"  [{i}] {ref}")
        print("Token IDs:", gen_ids)

        display = denormalize_image(item["image"]).permute(1, 2, 0).cpu().numpy()
        caption_block = "Generated:\n" + generated + "\n\nReferences:\n"
        caption_block += "\n".join(f"  • {r}" for r in references) or "  (none)"

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(display)
        ax.set_title(f"image_id={image_id}", fontsize=11)
        ax.axis("off")
        fig.text(
            0.5,
            0.02,
            caption_block,
            ha="center",
            va="bottom",
            fontsize=9,
            wrap=True,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 6},
        )
        fig.tight_layout(rect=[0, 0.12, 1, 1])

        if save_dir is not None:
            out_path = save_dir / f"preview_{image_id}.png"
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            print(f"Saved preview: {out_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        shown += 1
        if shown >= num_samples:
            break


def main() -> None:
    """Load checkpoint + val data, then preview generated captions with images."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ckpt", default="outputs/best.pt")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of unique validation images to caption and display",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Folder for preview PNGs (default: only print to terminal)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open matplotlib window for each preview (blocks until closed)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    resolve_path_fields(
        cfg,
        (
            "train_captions_json",
            "val_captions_json",
            "train_images_dir",
            "val_images_dir",
        ),
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg["device"] == "cuda" else "cpu"
    )
    ckpt_path = Path(args.ckpt).expanduser().resolve()
    st = torch.load(ckpt_path, map_location=device)

    if isinstance(st, dict) and isinstance(st.get("vocab"), list):
        vocab = vocab_from_itos(st["vocab"])
    else:
        tr_ids = select_image_ids(
            cfg["train_captions_json"], cfg.get("max_train_images")
        )
        vocab = build_vocab(
            cfg["train_captions_json"], cfg["vocab_min_freq"], image_ids=tr_ids
        )

    model = ImageCaptionerV1(
        len(vocab.itos),
        vocab.pad_id,
        cfg["word_dim"],
        cfg["hidden_dim"],
        cfg["max_regions"],
        cfg["question_dim"],
    ).to(device)
    model.load_state_dict(st.get("model", st), strict=True)
    model.eval()

    va_ids = select_image_ids(cfg["val_captions_json"], cfg.get("max_val_images"))
    ds = CocoCaptionDataset(
        cfg["val_images_dir"],
        cfg["val_captions_json"],
        vocab,
        cfg["max_caption_len"],
        cfg["val_image_filename_template"],
        image_ids=va_ids,
    )
    caps_by_image = load_caps(cfg["val_captions_json"])
    if va_ids is not None:
        caps_by_image = {i: caps_by_image[i] for i in va_ids if i in caps_by_image}

    save_dir = Path(args.save_dir).expanduser().resolve() if args.save_dir else None

    preview_caption_samples(
        model=model,
        dataset=ds,
        vocab=vocab,
        caps_by_image=caps_by_image,
        device=device,
        max_caption_len=cfg["max_caption_len"],
        num_samples=max(1, args.num_samples),
        save_dir=save_dir,
        show=args.show,
    )


if __name__ == "__main__":
    main()
