"""Hold-out evaluation for trained ``VQAModel`` checkpoints.

Paper alignment (*Image captioning improved visual question answering*)
-----------------------------------------------------------------------
Reports validation accuracy using **greedy decoding** of answer tokens (``a_ids=None`` path in
``VQAModel.forward``). Metrics mirror ``training.train.vqa_acc`` aggregation—cite evaluation
subsection alongside qualitative caption fusion discussion.

Split consistency
-----------------
Rebuilds vocabs from the **training** questions/annotations JSON files, then evaluates on the
official validation JSON and images so checkpoint vocab sizes match training.

CLI Examples
------------
::

    cd VQA
    python evaluation/evaluate.py --config configs/default.yaml --ckpt outputs/best.pt

Examples (shape sanity)
-----------------------
::

    logits = model(images, q_ids, a_ids=None, max_answer_len=cfg["max_answer_len"])
    pred = logits.argmax(dim=-1)  # (batch, time)

Notes
-----
Ensure ``captioner_ckpt`` / paths in YAML match those used during training; mismatched captioner
weights compromise fusion branch despite identical answer checkpoint loading.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.vqa_dataset import VQADataset, all_qids, build_vocabs, collate
from models.captioner_adapter import load_captioner
from models.vqa_model import VQAModel
from training.train import vqa_acc
from utils.common import load_config, resolve_path_fields


def main() -> None:
    """Load config + checkpoint, iterate loader, print mean batch accuracy."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ckpt", default="outputs/best.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    resolve_path_fields(
        cfg,
        (
            "train_questions_json",
            "train_annotations_json",
            "val_questions_json",
            "val_annotations_json",
            "train_images_dir",
            "val_images_dir",
            "captioner_project_root",
            "captioner_ckpt",
        ),
    )
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"] == "cuda" else "cpu")

    qv, av = build_vocabs(cfg["train_questions_json"], cfg["train_annotations_json"], cfg["vocab_min_freq"])
    va_qids = all_qids(cfg["val_questions_json"])
    ds = VQADataset(
        cfg["val_questions_json"],
        cfg["val_annotations_json"],
        cfg["val_images_dir"],
        cfg["val_image_filename_template"],
        qv,
        av,
        cfg["max_question_len"],
        cfg["max_answer_len"],
        qids=va_qids,
    )
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], collate_fn=collate)

    captioner = load_captioner(cfg, len(qv.itos), qv.pad_id, device)
    model = VQAModel(len(qv.itos), len(av.itos), qv.pad_id, captioner, cfg["word_dim"], cfg["hidden_dim"], cfg["question_dim"], cfg["max_regions"], cfg["fuse_mode"]).to(device)
    st = torch.load(Path(args.ckpt).expanduser().resolve(), map_location=device)
    model.load_state_dict(st.get("model", st), strict=False)
    model.eval()

    total = 0.0
    n = 0
    with torch.no_grad():
        for b in dl:
            logits = model(b["images"].to(device), b["q"].to(device), a_ids=None, max_answer_len=cfg["max_answer_len"])
            total += vqa_acc(logits.argmax(dim=-1), b["answers"], av)
            n += 1
    print(f"Validation VQA accuracy: {total / max(1, n):.4f}")


if __name__ == "__main__":
    main()
