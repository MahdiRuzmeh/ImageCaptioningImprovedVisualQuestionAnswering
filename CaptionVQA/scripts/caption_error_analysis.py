"""Why are the BLIP captions wrong? Error analysis on the 10,001 test questions (no GPU, no BLIP).

Finglish: baraye har soal-e test: caption-e cross-fitted, 10 javab-e ensan, va javab-e model haye VQA.
  - "strict right"  = hamoon ta'rif-e report: javab-e majority dar caption (yes/no: polarity-e dorost)
  - "soft"          = behtarin VQA score beyn-e javab haye ensan ke dar caption hast (synonym/javab-e dige-ye ensan)
Caption haye ghalat dar goroh ha: khondan-e matn/saat, shomaresh, rang, baghie; dalil-e khata; va inke model-e VQA
bedoon caption javab ro midoone ya na (captioner az khod-e VQA zaiftar?).

    python scripts/caption_error_analysis.py --out caption_error_analysis.md
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT.parent / "QuestionDependentImageCaptioner"))
from common import normalize_answer, tok, vqa_score  # noqa: E402
from metrics import answer_in_caption, caption_negated  # noqa: E402

READ = re.compile(r"\b(brand|name|names|named|written|write|writing|say|says|said|sign|signs|letter|letters|word|words|"
                  r"logo|read|team|company|airline|website|text|number on|numbers on|jersey|what time|clock|license|"
                  r"plate|label|title|flavor|model|make)\b")
COLORS = {"white", "black", "red", "blue", "green", "yellow", "brown", "gray", "grey", "orange", "pink", "purple",
          "silver", "gold", "tan", "beige", "maroon", "navy", "teal", "cream", "dark", "light"}
NUMS = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
        "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"}


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def group(r: dict, gt: str) -> str:
    q = r["question"].lower()
    if gt in ("yes", "no"):
        return "yes/no: is there / are there" if re.match(r"^(is|are) there\b", q) else "yes/no: other"
    if READ.search(q) or ":" in gt:
        return "open: read text / time"
    if r["answer_type"] == "number":
        return "open: count 0-3" if gt.isdigit() and int(gt) <= 3 else "open: count 4+ / other number"
    if "color" in q:
        return "open: color"
    return "open: other (objects, places, activities ...)"


def singular(ws):
    return [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in ws]


def analyse(rec: dict, cap: str):
    answers = rec["answers"]
    votes = Counter(normalize_answer(a) for a in answers)
    gt, nmaj = votes.most_common(1)[0]
    c = tok(cap)
    if gt in ("yes", "no"):
        right = caption_negated(c) == (gt == "no")
        return {"gt": gt, "votes": nmaj, "right": right, "soft": float(right),
                "reason": None if right else ("caption says no, answer yes" if gt == "yes" else "caption says yes, answer no")}
    right = bool(tok(gt)) and answer_in_caption(tok(gt), c)
    soft = max((vqa_score(a, answers) for a in votes if tok(a) and answer_in_caption(tok(a), c)), default=0.0)
    reason = None
    if not right:
        if soft > 0:
            reason = "another human answer is in the caption"
        elif tok(gt) and answer_in_caption(singular(tok(gt)), singular(c)):
            reason = "only singular/plural differs"
        elif rec["answer_type"] == "number" or gt in NUMS:
            reason = ("caption gives a different number" if any(w.isdigit() or w in NUMS for w in c)
                      else "caption gives no number")
        elif any(w in COLORS for w in tok(gt)):
            reason = "caption gives a different color" if any(w in COLORS for w in c) else "caption gives no color"
        else:
            reason = "caption names something else"
    return {"gt": gt, "votes": nmaj, "right": right, "soft": soft, "reason": reason}


def pct(x, n):
    return 100 * x / n if n else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default=str(ROOT / "outputs/blip_strong/split/test.json"))
    ap.add_argument("--captions", default=str(ROOT / "vqa_strong_crossfit/captions/captions_crossfit.json"))
    ap.add_argument("--baseline", default=str(ROOT / "vqa_b64_baseline/runs/nocap"))
    ap.add_argument("--caption-model", default=str(ROOT / "vqa_b64_crossfit_drop/runs/blip"))
    ap.add_argument("--out", default=str(ROOT / "caption_error_analysis.md"))
    args = ap.parse_args()

    recs = load(args.split)
    caps = load(args.captions)
    P = {}
    for name, d in (("base", args.baseline), ("cap", args.caption_model)):
        runs = [{p["question_id"]: p["score"] for p in load(f"{d}_s{s}/test_predictions.json")} for s in (0, 1)]
        P[name] = {q: statistics.mean(r[q] for r in runs) for q in runs[0]}

    rows = []
    for r in recs:
        a = analyse(r, caps[str(r["question_id"])])
        a.update(q=r["question_id"], question=r["question"], caption=caps[str(r["question_id"])],
                 group=group(r, a["gt"]), base=P["base"][r["question_id"]], capm=P["cap"][r["question_id"]])
        rows.append(a)

    # --- 1. per group: how often the caption is right, and how good VQA is without/with it ---
    order = ["open: color", "open: count 0-3", "open: count 4+ / other number", "open: read text / time",
             "open: other (objects, places, activities ...)", "yes/no: is there / are there", "yes/no: other"]
    by = defaultdict(list)
    for x in rows:
        by[x["group"]].append(x)
    md = ["# Why are the captions wrong? (10,001 test questions, cross-fitted BLIP-base captions)", "",
          "Caption right = the majority human answer is in the caption (yes/no: right polarity), as in the reports.",
          "Caption soft = best VQA score among the 10 human answers that appear in the caption (accepts other human answers).",
          "VQA columns: official accuracy, mean of 2 seeds (baseline = 1b-A, + captions = 1b-B with dropout).", "",
          "## 1. By question group", "",
          "| Group | n | Caption right | Caption soft | VQA without caption | VQA + caption | Gain |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    summary = {}
    for g in order + ["all"]:
        xs = rows if g == "all" else by[g]
        n = len(xs)
        s = {"n": n, "right": pct(sum(x["right"] for x in xs), n), "soft": 100 * statistics.mean(x["soft"] for x in xs),
             "base": 100 * statistics.mean(x["base"] for x in xs), "cap": 100 * statistics.mean(x["capm"] for x in xs)}
        summary[g] = s
        md.append(f"| {'**all**' if g == 'all' else g} | {n:,} | {s['right']:.1f} | {s['soft']:.1f} | {s['base']:.1f} | "
                  f"{s['cap']:.1f} | {s['cap'] - s['base']:+.1f} |")

    # --- 2. why open-question captions are wrong ---
    wrong_open = [x for x in rows if not x["right"] and x["gt"] not in ("yes", "no")]
    rc = Counter(x["reason"] for x in wrong_open)
    md += ["", f"## 2. Why open-question captions are wrong ({len(wrong_open):,} questions)", "",
           "| Reason | n | Share | VQA without caption is right (score ≥ 0.6) |", "|---|---:|---:|---:|"]
    reasons = {}
    for k, v in rc.most_common():
        xs = [x for x in wrong_open if x["reason"] == k]
        vqa_ok = pct(sum(x["base"] >= 0.6 for x in xs), len(xs))
        reasons[k] = {"n": v, "share": pct(v, len(wrong_open)), "base_right": vqa_ok}
        md.append(f"| {k} | {v:,} | {pct(v, len(wrong_open)):.1f}% | {vqa_ok:.1f}% |")
    md += ["", "Wrong open captions by group:", "", "| Group | wrong | of all wrong open |", "|---|---:|---:|"]
    gc = Counter(x["group"] for x in wrong_open)
    for g in order[:5]:
        md.append(f"| {g} | {gc[g]:,} | {pct(gc[g], len(wrong_open)):.1f}% |")

    # --- 3. yes/no ---
    yn = [x for x in rows if x["gt"] in ("yes", "no")]
    wrong_yn = [x for x in yn if not x["right"]]
    yc = Counter(x["reason"] for x in wrong_yn)
    md += ["", f"## 3. Yes/no captions with the wrong polarity ({len(wrong_yn):,} of {len(yn):,})", "",
           "| Error | n | VQA without caption is right |", "|---|---:|---:|"]
    for k, v in yc.most_common():
        xs = [x for x in wrong_yn if x["reason"] == k]
        md.append(f"| {k} | {v:,} | {pct(sum(x['base'] >= 0.6 for x in xs), len(xs)):.1f}% |")
    ans_yes = sum(x["gt"] == "yes" for x in yn)
    neg = sum(caption_negated(tok(x["caption"])) for x in yn)
    md.append(f"\nAnswer \"yes\" for {pct(ans_yes, len(yn)):.1f}% of yes/no questions; captions are negative for "
              f"{pct(neg, len(yn)):.1f}%.")

    # --- 4. human agreement ---
    md += ["", "## 4. Human agreement on the answer", "",
           "| Votes for the majority answer (of 10) | n | Caption right | VQA without caption |", "|---|---:|---:|---:|"]
    agree = {}
    for lo, hi, lab in ((8, 10, "8–10 (clear)"), (5, 7, "5–7"), (0, 4, "≤ 4 (ambiguous)")):
        xs = [x for x in rows if lo <= x["votes"] <= hi]
        agree[lab] = {"n": len(xs), "right": pct(sum(x["right"] for x in xs), len(xs)),
                      "base": 100 * statistics.mean(x["base"] for x in xs)}
        md.append(f"| {lab} | {len(xs):,} | {agree[lab]['right']:.1f} | {agree[lab]['base']:.1f} |")

    # --- 5. captioner vs VQA: who knows the answer? ---
    both = sum(x["right"] and x["base"] >= 0.6 for x in rows)
    cap_only = sum(x["right"] and x["base"] < 0.6 for x in rows)
    vqa_only = sum((not x["right"]) and x["base"] >= 0.6 for x in rows)
    neither = len(rows) - both - cap_only - vqa_only
    md += ["", "## 5. Captioner vs VQA without caption: who knows the answer?", "",
           "| | n | Share |", "|---|---:|---:|",
           f"| both | {both:,} | {pct(both, len(rows)):.1f}% |",
           f"| only the caption (new information for VQA) | {cap_only:,} | {pct(cap_only, len(rows)):.1f}% |",
           f"| only the VQA model (caption can only mislead) | {vqa_only:,} | {pct(vqa_only, len(rows)):.1f}% |",
           f"| neither | {neither:,} | {pct(neither, len(rows)):.1f}% |"]

    # --- 6. examples ---
    md += ["", "## 6. Examples of wrong open captions (first 3 per reason)", ""]
    for k in rc:
        md.append(f"**{k}**")
        md.append("")
        for x in [x for x in wrong_open if x["reason"] == k][:3]:
            md.append(f"- Q: {x['question']} — answer *{x['gt']}* — caption: \"{x['caption']}\"")
        md.append("")
    Path(args.out).write_text("\n".join(md) + "\n", encoding="utf-8")
    Path(args.out).with_suffix(".json").write_text(json.dumps(
        {"groups": summary, "open_reasons": reasons, "agreement": agree,
         "who_knows": {"both": both, "caption_only": cap_only, "vqa_only": vqa_only, "neither": neither}}, indent=1),
        encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
