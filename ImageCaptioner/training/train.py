import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torch import amp
from torch.optim import Adamax
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from datasets.coco_caption_dataset import CocoCaptionDataset, build_vocab, collate, load_caps, split_ids
from models.captioner_v1 import ImageCaptionerV1
from utils.common import load_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"] == "cuda" else "cpu")
    ids = list(load_caps(cfg["dataset_root"]).keys())
    tr_ids, va_ids = split_ids(ids, seed=cfg["seed"])
    vocab = build_vocab(cfg["dataset_root"], tr_ids, cfg["vocab_min_freq"])

    tr = CocoCaptionDataset(cfg["dataset_root"], tr_ids, vocab, cfg["max_caption_len"])
    va = CocoCaptionDataset(cfg["dataset_root"], va_ids, vocab, cfg["max_caption_len"])
    tr_loader = DataLoader(tr, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], collate_fn=collate)
    va_loader = DataLoader(va, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], collate_fn=collate)

    model = ImageCaptionerV1(len(vocab.itos), vocab.pad_id, cfg["word_dim"], cfg["hidden_dim"], cfg["max_regions"], cfg["question_dim"]).to(device)
    opt = Adamax(model.parameters(), lr=cfg["learning_rate"])
    sch = StepLR(opt, step_size=cfg["lr_decay_every"], gamma=cfg["lr_decay_factor"])
    scaler = amp.GradScaler("cuda", enabled=cfg["use_amp"] and device.type=="cuda")
    crit = nn.CrossEntropyLoss(ignore_index=0)

    best = 1e9
    out = Path(cfg["save_dir"])
    out.mkdir(parents=True, exist_ok=True)

    for ep in range(1, cfg["epochs"]+1):
        model.train()
        tr_loss = 0.0
        opt.zero_grad(set_to_none=True)
        for i,b in enumerate(tr_loader):
            images = b["images"].to(device)
            caps = b["captions"].to(device)
            with amp.autocast("cuda", enabled=cfg["use_amp"] and device.type=="cuda"):
                logits = model.forward_train(images, caps)
                loss = crit(logits.reshape(-1, logits.size(-1)), caps[:,1:].reshape(-1))
            scaler.scale(loss / cfg["grad_accum_steps"]).backward()
            if (i+1) % cfg["grad_accum_steps"] == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            tr_loss += float(loss.item())

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for b in va_loader:
                images = b["images"].to(device)
                caps = b["captions"].to(device)
                logits = model.forward_train(images, caps)
                loss = crit(logits.reshape(-1, logits.size(-1)), caps[:,1:].reshape(-1))
                va_loss += float(loss.item())

        sch.step()
        tr_loss /= max(1,len(tr_loader))
        va_loss /= max(1,len(va_loader))
        print(f"Epoch {ep}: train={tr_loss:.4f} val={va_loss:.4f}")
        state={"model":model.state_dict(),"vocab":vocab.itos,"config":cfg}
        torch.save(state, out / "last.pt")
        if va_loss < best:
            best=va_loss
            torch.save(state, out / "best.pt")

if __name__ == "__main__":
    main()
