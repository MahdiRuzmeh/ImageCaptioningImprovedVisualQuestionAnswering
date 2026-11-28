"""Analyse the human review: rates per criterion and group (95% Wilson intervals), the population-weighted error
rate, and agreement between two reviewers on the overlap items (Cohen's kappa).

A caption counts as CORRECT when criteria 1-3 all hold (answer stated correctly, subject/relation kept, no added
fact). Grammar (4) and visual answerability (5) are reported separately.

    python audit/human_review/analyse_review.py --labels review_labels_A.json [review_labels_B.json] \
        [--adjudicated review_labels_final.json]
The primary labels are the adjudicated file if given, else the first reviewer's file.
Output: review_results.md / review_results.json next to this script.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
CRIT = ["answer", "subject", "facts", "grammar", "visual"]
NAMES = {"answer": "1 answer stated correctly", "subject": "2 subject and relation kept", "facts": "3 no added fact",
         "grammar": "4 grammatical", "visual": "5 visually answerable", "correct": "correct (1 and 2 and 3)"}


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def kappa(a, b):
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def complete(lab):
    return all(lab.get(c) in (0, 1) for c in CRIT)


def with_correct(lab):
    return {**lab, "correct": int(lab["answer"] == 1 and lab["subject"] == 1 and lab["facts"] == 1)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--adjudicated")
    args = p.parse_args()
    key = json.loads((HERE / "sample_key.json").read_text(encoding="utf-8"))
    meta, group = key["meta"], {k["item"]: k["group"] for k in key["items"]}
    files = [json.loads(Path(f).read_text(encoding="utf-8")) for f in args.labels]
    primary = json.loads(Path(args.adjudicated).read_text(encoding="utf-8")) if args.adjudicated else files[0]
    labs = {int(i): with_correct(l) for i, l in primary["labels"].items() if complete(l)}
    missing = len(group) - len(labs)

    res = {"meta": meta, "primary": primary.get("reviewer"), "complete_items": len(labs), "missing_items": missing,
           "groups": {}, "weighted": {}, "agreement": {}}
    pop = meta["population"]
    total = sum(pop.values())
    for c in CRIT + ["correct"]:
        res["groups"][c] = {}
        for g in ("unflagged", "flagged"):
            vals = [l[c] for i, l in labs.items() if group[i] == g]
            res["groups"][c][g] = dict(zip(("rate", "lo", "hi"), wilson(sum(vals), len(vals)))) | {"n": len(vals), "k": sum(vals)}
        # population-weighted rate: e = sum_g (N_g / N) e_g ; stratified variance sum_g (N_g/N)^2 e_g(1-e_g)/n_g
        est, var = 0.0, 0.0
        for g in ("unflagged", "flagged"):
            r = res["groups"][c][g]
            if r["n"]:
                w = pop[g] / total
                est += w * r["rate"]
                var += w * w * r["rate"] * (1 - r["rate"]) / r["n"]
        res["weighted"][c] = {"rate": est, "lo": est - 1.96 * math.sqrt(var), "hi": est + 1.96 * math.sqrt(var)}

    if len(files) > 1:
        a = {int(i): l for i, l in files[0]["labels"].items() if complete(l)}
        b = {int(i): l for i, l in files[1]["labels"].items() if complete(l)}
        both = sorted(set(a) & set(b))
        for c in CRIT + ["correct"]:
            x = [with_correct(a[i])[c] for i in both]
            y = [with_correct(b[i])[c] for i in both]
            res["agreement"][c] = {"n": len(both), "agree": sum(u == v for u, v in zip(x, y)) / max(1, len(both)),
                                   "kappa": kappa(x, y),
                                   "disagreements": [i for i, u, v in zip(both, x, y) if u != v]}

    L = ["# Human review of the question-dependent captions", "",
         f"Sample: {meta['sample']} from population {pop} (seed {meta['seed']}); complete items {len(labs)}, "
         f"missing {missing}; primary labels: {res['primary']}.", "",
         "| Criterion | Unflagged: rate [95% CI] (k/n) | Flagged: rate [95% CI] (k/n) | Weighted population rate [95% CI] |",
         "|---|---|---|---|"]
    for c in CRIT + ["correct"]:
        u, f, w = res["groups"][c]["unflagged"], res["groups"][c]["flagged"], res["weighted"][c]
        L.append(f"| {NAMES[c]} | {100*u['rate']:.1f} [{100*u['lo']:.1f}, {100*u['hi']:.1f}] ({u['k']}/{u['n']}) | "
                 f"{100*f['rate']:.1f} [{100*f['lo']:.1f}, {100*f['hi']:.1f}] ({f['k']}/{f['n']}) | "
                 f"{100*w['rate']:.1f} [{100*w['lo']:.1f}, {100*w['hi']:.1f}] |")
    e = res["weighted"]["correct"]
    L += ["", f"**Estimated caption error rate of the training data** (1 − correct, weights {pop['unflagged']}/{total} and "
          f"{pop['flagged']}/{total}): **{100*(1-e['rate']):.1f}%** [{100*(1-e['hi']):.1f}, {100*(1-e['lo']):.1f}]."]
    if res["agreement"]:
        L += ["", "| Criterion | Overlap items | Agreement | Cohen's kappa | Items to resolve |", "|---|--:|--:|--:|---|"]
        for c, a in res["agreement"].items():
            L.append(f"| {NAMES[c]} | {a['n']} | {100*a['agree']:.1f}% | {a['kappa']:.2f} | {', '.join(map(str, a['disagreements'][:20]))} |")
    (HERE / "review_results.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (HERE / "review_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
