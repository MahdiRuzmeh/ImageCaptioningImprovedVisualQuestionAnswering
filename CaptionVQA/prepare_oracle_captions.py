"""Oracle caption features: the gold answer is written into every caption (upper-bound analysis).

Finglish:
    Har (image, soal): javab-e aksariat (normalize_answer) + caption-e BUTD captioner
    (``captions_{split}_butd.pt``) → "yellow the boat is red". Hame 3 beam javab ro avval
    daran ta truncate nashe. Javab az annotation miyad → test-time CHEAT ast; faghat baraye
    andaze-giri-e saghf: "age captioner kamel bood, VQA cheghadr behtar mishod?"
    ``hidden`` / ``scores`` ba teacher forcing-e hamoon captioner (mesl-e prepare_human_captions).
    Kalame haye kharej az vocab-e captioner → ``<unk>`` (oracle-coverage gozaresh mishe).

    python prepare_oracle_captions.py --config configs/small_butd.yaml
    → work_dir/features/captions_{train,dev,test}_oracle.pt
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm

from common import load_config, normalize_answer, read_json, tok, write_json
from prepare_features import SPLITS, captioner_visual_cache, encode_question, load_frozen_captioner, pad_batch, strip_bos_eos
from prepare_human_captions import teacher_force

BASE_SET = "butd"


def gold_answer(answers) -> str:
    return Counter(normalize_answer(a) for a in answers).most_common(1)[0][0]


@torch.no_grad()
def caption_split(cfg: dict, name: str, captioner, vocab, q_vocab, device, cache_dir: str) -> None:
    work = Path(cfg["work_dir"])
    out = work / "features" / f"captions_{name}_oracle.pt"
    if out.exists():
        print(f"[oracle] {out} exists, skip")
        return
    recs = read_json(work / "split" / f"{name}.json")
    base = torch.load(work / "features" / f"captions_{name}_{BASE_SET}.pt")
    assert base["question_ids"] == [r["question_id"] for r in recs]
    L = int(cfg["caption_max_len"]) - 1
    unk = vocab.stoi["<unk>"]
    all_tok, all_score, all_hid = [], [], []
    covered = 0
    bs = 128
    for i in tqdm(range(0, len(recs), bs), desc=f"oracle/{name}"):
        chunk = recs[i : i + bs]
        ans_ids = [[vocab.stoi.get(w, unk) for w in tok(gold_answer(r["answers"]))] for r in chunk]
        covered += sum(1 for a in ans_ids if a and unk not in a)
        q = pad_batch([encode_question(r["question"], q_vocab, int(cfg["max_question_len"])) for r in chunk]).to(device)
        iids = torch.tensor([r["image_id"] for r in chunk], device=device)
        dummy = torch.zeros((len(chunk), 3, 1, 1), device=device)
        s = captioner._encode(dummy, q, iids, cache_dir, False)
        beams, scores = [], []
        for k in range(base["tokens"].size(1)):
            rows = []
            for j, a in enumerate(ans_ids):
                cap = [t for t in base["tokens"][i + j, k].tolist() if t != 0]
                rows.append([1] + (a + cap)[:L] + [2])
            seqs = pad_batch(rows).to(device)
            sc, h = teacher_force(captioner, s, seqs)
            beams.append(seqs)
            scores.append(sc)
            if k == 0:  # Finglish: decoder state faghat baraye beam 1 (mesl-e generated)
                hid = h[:, 1:]
                hid = torch.nn.functional.pad(hid, (0, 0, 0, max(0, L - hid.size(1))))[:, :L]
        T = max(b.size(1) for b in beams)
        seqs = torch.stack([torch.nn.functional.pad(b, (0, T - b.size(1))) for b in beams], 1)
        tokens = strip_bos_eos(seqs, L)
        hid = hid * (tokens[:, 0] != 0).unsqueeze(-1)
        all_tok.append(tokens.short().cpu())
        all_score.append(torch.stack(scores, 1).float().cpu())
        all_hid.append(hid.half().cpu())
    tokens = torch.cat(all_tok)
    torch.save({
        "question_ids": [r["question_id"] for r in recs],
        "tokens": tokens,
        "scores": torch.cat(all_score),
        "hidden": torch.cat(all_hid),
    }, out)
    sample = []
    for k in range(0, len(recs), max(1, len(recs) // 30)):
        caps = [" ".join(vocab.itos[t] for t in row.tolist() if t > 0) for row in tokens[k]]
        sample.append({"question": recs[k]["question"], "answers": recs[k]["answers"][:3], "captions": caps})
    write_json(work / "features" / f"captions_{name}_oracle_sample.json", sample)
    print(f"[oracle] wrote {out} tokens={tuple(tokens.shape)} answer-in-vocab={covered / len(recs):.3f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/small_butd.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    captioner, vocab, q_vocab, cap_cfg = load_frozen_captioner(cfg["captioner_ckpt"], device)
    cache_dir = captioner_visual_cache(cap_cfg)
    for name in SPLITS:
        caption_split(cfg, name, captioner, vocab, q_vocab, device, cache_dir)


if __name__ == "__main__":
    main()
