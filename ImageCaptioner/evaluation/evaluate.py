"""Minimal caption checkpoint smoke test after stage-one training.

Purpose
-------
Not a full MSCOCO caption benchmark—loads ``best.pt`` / ``last.pt``, runs **one** greedy decoding
batch from validation split, prints token IDs. Use BLEU/CIDEr notebooks for thesis-quality metrics.

Paper reference (*Image captioning improved visual question answering*)
------------------------------------------------------------------------
Point readers from qualitative subsection here; quantitative caption scores belong in evaluation
tables referencing MSCOCO caption metrics **before** reporting VQA uplift.

CLI Examples
------------
::

    cd ImageCaptioner
    python evaluation/evaluate.py --config configs/default.yaml --ckpt outputs/best.pt

Examples (decode manually)
--------------------------
::

    ids = pred[0].tolist()
    words = [vocab.itos[i] for i in ids if i > 2]

Notes
-----
Greedy decoding ignores beam search used in some papers—extend script if thesis compares decoding
strategies.
"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from datasets.coco_caption_dataset import (
    CocoCaptionDataset,
    build_vocab,
    collate,
    select_image_ids,
)
from models.captioner_v1 import ImageCaptionerV1
from utils.common import load_config, resolve_path_fields


def main() -> None:
    """Instantiate loader + model, emit preview tokens."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ckpt", default="outputs/best.pt")
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
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"] == "cuda" else "cpu")

    tr_ids = select_image_ids(
        cfg["train_captions_json"], cfg.get("max_train_images"))
    va_ids = select_image_ids(cfg["val_captions_json"], cfg.get("max_val_images"))
    vocab = build_vocab(
        cfg["train_captions_json"], cfg["vocab_min_freq"], image_ids=tr_ids)
    ds = CocoCaptionDataset(
        cfg["val_images_dir"],
        cfg["val_captions_json"],
        vocab,
        cfg["max_caption_len"],
        cfg["val_image_filename_template"],
        image_ids=va_ids,
    )
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], collate_fn=collate)

    model = ImageCaptionerV1(len(vocab.itos), vocab.pad_id, cfg["word_dim"], cfg["hidden_dim"], cfg["max_regions"], cfg["question_dim"]).to(device)
    ckpt_path = Path(args.ckpt).expanduser().resolve()
    st = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(st.get("model", st), strict=False)
    model.eval()

    with torch.no_grad():
        for b in dl:
            pred = model.generate_caption(b["images"].to(device), max_len=cfg["max_caption_len"])
            print("Preview generated caption token IDs:", pred[0].tolist())
            break


if __name__ == "__main__":
    main()
