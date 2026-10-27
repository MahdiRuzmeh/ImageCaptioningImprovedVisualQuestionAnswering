"""How often do the frozen captioner's captions contain a correct answer? (per answer type)

    python caption_stats.py --config configs/small.yaml --split test
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from common import load_config, normalize_answer, read_json, soft_targets


def ngrams(words, n_max=3):
    return {normalize_answer(" ".join(words[s : s + n])) for s in range(len(words))
            for n in range(1, n_max + 1) if s + n <= len(words)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/small.yaml")
    p.add_argument("--split", default="test")
    p.add_argument("--captions", choices=["generated", "human"], default="generated")
    args = p.parse_args()
    cfg = load_config(args.config)
    work = Path(cfg["work_dir"])
    recs = read_json(work / "split" / f"{args.split}.json")
    suffix = "" if args.captions == "generated" else f"_{args.captions}"
    cap = torch.load(work / "features" / f"captions_{args.split}{suffix}.pt")
    itos = torch.load(cfg["captioner_ckpt"], map_location="cpu", weights_only=False)["vocab"]
    stats = defaultdict(lambda: defaultdict(float))
    for r, toks in zip(recs, cap["tokens"]):
        st = soft_targets(r["answers"])
        beams = [[itos[t] for t in row.tolist() if t != 0] for row in toks]
        g1 = ngrams(beams[0])
        gall = set().union(*(ngrams(b) for b in beams))
        for t in (r["answer_type"], "all"):
            s = stats[t]
            s["n"] += 1
            s["beam1_has_answer"] += max((st.get(a, 0) for a in g1), default=0)
            s["any_beam_has_answer"] += max((st.get(a, 0) for a in gall), default=0)
    print(f"| Answer type | n | Best answer in beam-1 caption (soft acc) | Best answer in any of 3 beams |")
    print("|---|---|---|---|")
    for t in ("all", "yes/no", "number", "other"):
        s = stats[t]
        print(f"| {t} | {int(s['n'])} | {100*s['beam1_has_answer']/s['n']:.1f} | {100*s['any_beam_has_answer']/s['n']:.1f} |")


if __name__ == "__main__":
    main()
