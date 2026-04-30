import argparse

import torch
from torch.utils.data import DataLoader

from datasets.coco_caption_dataset import CocoCaptionDataset, build_vocab, collate, load_caps, split_ids
from models.captioner_v1 import ImageCaptionerV1
from utils.common import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ckpt", default="outputs/best.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"] == "cuda" else "cpu")

    ids = list(load_caps(cfg["dataset_root"]).keys())
    tr, va = split_ids(ids, seed=cfg["seed"])
    vocab = build_vocab(cfg["dataset_root"], tr, cfg["vocab_min_freq"])
    ds = CocoCaptionDataset(cfg["dataset_root"], va, vocab, cfg["max_caption_len"])
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], collate_fn=collate)

    model = ImageCaptionerV1(len(vocab.itos), vocab.pad_id, cfg["word_dim"], cfg["hidden_dim"], cfg["max_regions"], cfg["question_dim"]).to(device)
    st = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(st.get("model", st), strict=False)
    model.eval()

    with torch.no_grad():
        for b in dl:
            pred = model.generate_caption(b["images"].to(device), max_len=cfg["max_caption_len"])
            print("Preview generated caption token IDs:", pred[0].tolist())
            break

if __name__ == "__main__":
    main()
