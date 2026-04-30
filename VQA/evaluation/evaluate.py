import argparse

import torch
from torch.utils.data import DataLoader

from datasets.vqa_dataset import VQADataset, all_qids, build_vocabs, collate, split_qids
from models.captioner_adapter import load_captioner
from models.vqa_model import VQAModel
from training.train import vqa_acc
from utils.common import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ckpt", default="outputs/best.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"] == "cuda" else "cpu")

    qids = all_qids(cfg["dataset_root"])
    tr, va = split_qids(qids, seed=cfg["seed"])
    qv, av = build_vocabs(cfg["dataset_root"], tr, cfg["vocab_min_freq"])
    ds = VQADataset(cfg["dataset_root"], va, qv, av, cfg["max_question_len"], cfg["max_answer_len"])
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], collate_fn=collate)

    captioner = load_captioner(cfg, len(qv.itos), qv.pad_id, device)
    model = VQAModel(len(qv.itos), len(av.itos), qv.pad_id, captioner, cfg["word_dim"], cfg["hidden_dim"], cfg["question_dim"], cfg["max_regions"], cfg["fuse_mode"]).to(device)
    st = torch.load(args.ckpt, map_location=device)
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
