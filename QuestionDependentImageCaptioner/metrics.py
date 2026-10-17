"""Caption metrics (BLEU-1..4 + CIDEr-D) — self-contained, no extra installs.

Finglish — chera in file?
    token accuracy (teacher forcing) grounding ro nemibine; metric vagheie caption
    BLEU/CIDEr roye caption haye **generate shode** hast. Inja az 0 implement shode
    (pycocoevalcap MISSING bood roye in env / Kaggle) ta repo portable bemune.
    Vorudi: predictions (str) + references (list[str] har image). Khorouji: dict score.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Sequence

import numpy as np


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    """Finglish: hame n-gram haye yek jomle ro shomaresh mikone (Counter)."""
    return Counter(
        tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
    )


# ---------------------------------------------------------------------------
# BLEU (corpus-level, BLEU-1..4 ba brevity penalty — closest ref length)
# ---------------------------------------------------------------------------
def corpus_bleu(
    hyps: List[List[str]], refs: List[List[List[str]]], max_n: int = 4
) -> Dict[str, float]:
    """Finglish — corpus BLEU:
        Baraye har order n (1..4) clipped n-gram match ha ro roye kol corpus jam mizanim,
        precision hesab mikonim, bad geometric mean + brevity penalty.
        hyps[i]=token haye pred, refs[i]=list az reference token-list baraye image i.
    """
    clipped = [0] * max_n
    totals = [0] * max_n
    hyp_len_total = 0
    ref_len_total = 0

    for hyp, ref_list in zip(hyps, refs):
        hyp_len_total += len(hyp)
        # closest reference length (BLEU brevity penalty convention)
        ref_len_total += min(
            (len(r) for r in ref_list),
            key=lambda rl: (abs(rl - len(hyp)), rl),
            default=0,
        )
        for n in range(1, max_n + 1):
            hyp_ng = _ngrams(hyp, n)
            totals[n - 1] += max(0, len(hyp) - n + 1)
            if not hyp_ng:
                continue
            max_ref = Counter()
            for r in ref_list:
                for g, c in _ngrams(r, n).items():
                    if c > max_ref[g]:
                        max_ref[g] = c
            for g, c in hyp_ng.items():
                clipped[n - 1] += min(c, max_ref[g])

    precisions = []
    for n in range(max_n):
        precisions.append(clipped[n] / totals[n] if totals[n] > 0 else 0.0)

    # brevity penalty
    if hyp_len_total == 0:
        bp = 0.0
    elif hyp_len_total > ref_len_total:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len_total / max(1, hyp_len_total))

    out: Dict[str, float] = {}
    log_sum = 0.0
    for n in range(1, max_n + 1):
        p = precisions[n - 1]
        # cumulative BLEU-n (geometric mean of p_1..p_n)
        log_sum += math.log(p) if p > 0 else -1e9
        out[f"BLEU-{n}"] = bp * math.exp(log_sum / n) if p > 0 else 0.0
    return out


# ---------------------------------------------------------------------------
# CIDEr-D (TF-IDF n-gram, gaussian length penalty — sigma=6)
# ---------------------------------------------------------------------------
def _counts2vec(
    cnts: Counter, df: Dict[tuple, float], ref_len: float, n: int
):
    """Finglish: count n-gram ha → bordar TF-IDF + norm + tool jomle (per CIDEr ref impl)."""
    vec = [defaultdict(float) for _ in range(n)]
    norm = [0.0] * n
    length = 0
    for ng, tf in cnts.items():
        nn = len(ng) - 1
        idf = ref_len - math.log(max(1.0, df.get(ng, 0.0)))
        vec[nn][ng] = float(tf) * idf
        norm[nn] += vec[nn][ng] ** 2
        if nn == 1:
            length += tf
    norm = [math.sqrt(v) for v in norm]
    # length: tedad unigram (n=1) ham hesab mikonim
    length = sum(cnts[g] for g in cnts if len(g) == 1)
    return vec, norm, length


def _sim(vh, vr, nh, nr, lh, lr, n: int, sigma: float = 6.0) -> np.ndarray:
    """Finglish: shabahat cosine clipped beyn cand/ref + jarimه tool (gaussian)."""
    delta = float(lh - lr)
    val = np.zeros(n)
    for nn in range(n):
        for ng, c in vh[nn].items():
            val[nn] += min(c, vr[nn].get(ng, 0.0)) * vr[nn].get(ng, 0.0)
        if nh[nn] != 0 and nr[nn] != 0:
            val[nn] /= nh[nn] * nr[nn]
        val[nn] *= math.exp(-(delta ** 2) / (2 * sigma ** 2))
    return val


def cider_d(
    hyps: List[List[str]], refs: List[List[List[str]]], n: int = 4
) -> float:
    """Finglish — CIDEr-D:
        Aval document-frequency har n-gram ro az reference ha mishmarim (har image=1 doc).
        Bad har candidate ro ba reference hash TF-IDF cosine misanjim + gaussian length penalty.
        Score nahayi = 10 × miangin (roye image va order ha). Standard captioning metric.
    """
    df: Dict[tuple, float] = defaultdict(float)
    for ref_list in refs:
        seen = set()
        for r in ref_list:
            for nn in range(1, n + 1):
                seen.update(_ngrams(r, nn).keys())
        for g in seen:
            df[g] += 1.0
    ref_len = math.log(max(1, len(refs)))

    scores = []
    for hyp, ref_list in zip(hyps, refs):
        vh, nh, lh = _counts2vec(
            sum((_ngrams(hyp, k) for k in range(1, n + 1)), Counter()), df, ref_len, n
        )
        acc = np.zeros(n)
        for r in ref_list:
            vr, nr, lr = _counts2vec(
                sum((_ngrams(r, k) for k in range(1, n + 1)), Counter()), df, ref_len, n
            )
            acc += _sim(vh, vr, nh, nr, lh, lr, n)
        acc /= max(1, len(ref_list))
        scores.append(10.0 * float(np.mean(acc)))
    return float(np.mean(scores)) if scores else 0.0


def compute_caption_metrics(
    hyps: List[List[str]], refs: List[List[List[str]]]
) -> Dict[str, float]:
    """Finglish: BLEU-1..4 + CIDEr ro yekja hesab mikone va dict barmigardune."""
    out = corpus_bleu(hyps, refs, max_n=4)
    out["CIDEr"] = cider_d(hyps, refs, n=4)
    return out
