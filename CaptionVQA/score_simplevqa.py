"""Score a SimpleVQA checkpoint on the CaptionVQA test split with the shared VQA v2 metric.

Run from SimpleVQA/ (its config paths are relative to it):
    python ../CaptionVQA/score_simplevqa.py --config configs/caption_vqa_small.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import read_json, vqa_score, write_json  # noqa: E402
sys.path.remove(str(HERE))  # CaptionVQA/models.py != captioner "models" package

SIMPLE_VQA = HERE.parent / "SimpleVQA"
sys.path.insert(0, str(SIMPLE_VQA))
import train as svqa  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/caption_vqa_small.yaml")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--split", default="test")
    args = p.parse_args()
    cfg = svqa.load_config(args.config)
    split_json = HERE / "outputs" / "small" / "split" / f"{args.split}.json"
    cfg["val_qids_json"] = str(split_json)
    svqa.resolve_path_fields(cfg, svqa.PATH_KEYS)
    cfg["num_workers"] = 2  # fit 16 GB RAM (each worker copies the dataset)
    device = torch.device("cuda")

    q_vocab, a_vocab, _, loader = svqa.build_loaders(cfg)
    model, _ = svqa.build_vqa_model(cfg, q_vocab, a_vocab, device)
    ckpt = Path(args.ckpt or Path(cfg["save_dir"]) / "best.pt")
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    # Finglish: vocab haye checkpoint (az train qids) — bayad ba build_loaders yeki bashan
    assert list(state.get("a_vocab", a_vocab.itos)) == a_vocab.itos
    model.eval()

    recs = {r["question_id"]: r for r in read_json(split_json)}
    samples = loader.dataset.samples
    preds = []
    k = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"SimpleVQA/{args.split}"):
            logits = model(
                batch["images"].to(device), batch["q"].to(device), a_ids=None,
                max_answer_len=int(cfg["max_answer_len"]), image_ids=batch["image_ids"].to(device),
                region_cache_dir=svqa.region_cache_dir_for_split(cfg, "val"),
                global_cache_dir=svqa.global_cache_dir_for_split(cfg, "val"),
                save_cache=True, q_cap_ids=svqa.batch_q_cap(batch, device),
            )
            for row in logits.argmax(-1).tolist():
                preds.append((samples[k]["qid"], svqa.decode_answer_ids(row, a_vocab)))
                k += 1

    by_type, scores, out = {}, [], []
    for qid, text in preds:
        r = recs[qid]
        sc = vqa_score(text, r["answers"])
        scores.append(sc)
        by_type.setdefault(r["answer_type"], []).append(sc)
        out.append({"question_id": qid, "question": r["question"], "pred": text, "score": sc})
    res = {"name": "SimpleVQA", "checkpoint_epoch": state.get("epoch"), "split": args.split,
           "acc": 100 * sum(scores) / len(scores), "n": len(scores)}
    res.update({f"acc_{t}": 100 * sum(v) / len(v) for t, v in sorted(by_type.items())})
    run_dir = HERE / "outputs" / "small" / "runs" / "SimpleVQA"
    write_json(run_dir / f"results_{args.split}.json", res)
    write_json(run_dir / f"{args.split}_predictions.json", out)
    print(res)


if __name__ == "__main__":
    main()
