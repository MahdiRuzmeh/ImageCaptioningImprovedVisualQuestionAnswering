"""Evidence for the manuscript (no GPU): data flow, cross-fitting proof, answer vocabulary, 10,000 vs 10,001,
image-clustered bootstrap on the 10k diagnostic set, official single-seed test-dev statistics.

Finglish: hame chiz az file haye mojood (QD JSON, folds JSON, split ha, score ha, natije haye EvalAI) hesab mishe.
Khorooji: ../manuscript/evidence/paper_evidence.{json,md}

    python scripts/paper_evidence.py
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]          # CaptionVQA
REPO = ROOT.parent
OUT = REPO / "manuscript" / "evidence"
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))
from common import normalize_answer, vqa_score  # noqa: E402

QD = REPO / "dataset" / "vqa_v2_question_dependent_captions_train2014.json"
FOLDS = ROOT / "data" / "crossfit_folds.json"
SPLIT = ROOT / "outputs" / "blip_strong" / "split"
RUNS = {"base": ("MCAN, no captions", "vqa_b64_baseline/runs/nocap"),
        "blip_base": ("MCAN + BLIP-base captions + dropout", "vqa_b64_crossfit_drop/runs/blip"),
        "large": ("MCAN + BLIP-large captions + dropout", "vqa_large_drop/runs/blip")}
TESTDEV = {"captions": ["vqa_final_t3/vqa_final/runs/blip_s0", "vqa_final_t3/vqa_final/runs/blip_s1"],
           "baseline": ["vqa_final_t4/vqa_final_baseline/runs/nocap_s0", "vqa_final_t4/vqa_final_baseline/runs/nocap_s1"]}
ENSEMBLE = {"captions": {"test-dev": "vqa_final_t3/vqa_final/Ensemble-VQA-With-Large-Blip-Test-Standard-Result-EvalAI.json"},
            "baseline": {"test-dev": "vqa_final_t4/vqa_final_baseline/Ensemble-VQA-BaseLine-Test-Standard-Result-EvalAI.json"}}
TYPES = ("overall", "yes/no", "number", "other")
B = 5000


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def majority(answers):
    return Counter(normalize_answer(a) for a in answers).most_common(1)[0][0]


def snap(rows, i):
    i = max(0, min(i, len(rows)))
    while 0 < i < len(rows) and rows[i]["image_id"] == rows[i - 1]["image_id"]:
        i += 1
    return i


def evalai(p):
    d = {}
    for x in load(p):
        d.update(x)
    return d


# ---------------------------------------------------------------- data flow
def data_flow(info, rows):
    rc, fix = info["question_reclassifier"], info["caption_error_fix"]
    pr = fix["progress"]
    verdict = Counter(r["caption_quality_verdict"] for r in rows)
    ok = [r for r in rows if r["caption_quality_verdict"] == "ok"]
    return {
        "train2014_questions": info["input_count"],
        "first_pass": {"classifier_model": info["question_classifier"]["model"],
                       "classifier_prompt": info["question_classifier"]["prompt_version"],
                       "generator_model": info["llm"]["model"], "generation_prompt": info["versions"]["generation_prompt"],
                       "blacklist_not_visual": info["question_classifier"]["label_counts"]["BLACKLIST_NDV"],
                       "ocr_excluded": info["ocr_excluded_count"], "duplicates": info["duplicate_count"],
                       "dropped_empty": info["dropped_empty_count"],
                       "validation_failures": info["validation_failure_count"],
                       "caption_records": rc["rows_before"]},
        "reclassification": {"model": rc["model"], "prompt": rc["prompt_version"], "rows_before": rc["rows_before"],
                             "rows_after": rc["rows_after"], "labels": rc["label_counts"]},
        "screening_and_repair": {"model": fix["model"], "judge_prompt": fix["judge_prompt_version"],
                                 "regen_prompt": fix["regen_prompt_version"], "regen_rounds": fix["regen_rounds"],
                                 "regen_temperature": fix["regen_temperature"],
                                 "judged": pr["rows_judged"], "initial": pr["original_verdict_counts"],
                                 "initial_wrong_reasons": pr["original_wrong_reasons"],
                                 "regenerated": pr["regenerated"], "fixed": pr["fixed"],
                                 "still_wrong": pr["still_wrong"], "final": pr["final_verdict_counts"],
                                 "suspicious_flag": pr["suspicious"],
                                 "suspicious_reasons": pr["suspicious_reason_counts"]},
        "file_verdicts": dict(verdict),
        "captioner_training_rows (verdict == ok)": len(ok),
        "ok_rows_with_suspicious_flag": sum(r["suspicious"] for r in ok),
        "ok_images": len({r["image_id"] for r in ok}),
        "wrong_share_of_screened_pct": 100 * pr["still_wrong"] / pr["rows_judged"],
    }


# ---------------------------------------------------------------- cross-fitting
def crossfit(rows):
    folds = load(FOLDS)
    tr, va = folds["train2014"], folds["val2014"]
    ok = [r for r in rows if r["caption_quality_verdict"] == "ok"]
    res = {"fold_seed": folds["seed"],
           "train2014_images_per_fold": dict(Counter(tr.values())),
           "val2014_images_per_fold": dict(Counter(va.values())),
           "train_val_image_overlap": len(set(tr) & set(va)),
           "qd_images_without_fold": sum(1 for r in ok if str(r["image_id"]) not in tr),
           "qd_images_in_val2014": sum(1 for r in ok if str(r["image_id"]) in va),
           "folds": {}}
    for k in (0, 1):
        fr = [r for r in ok if tr[str(r["image_id"])] == k]          # make_fold_qd.py
        n = len(fr)
        a, b = snap(fr, 0), snap(fr, n - 1500)
        train_imgs = {r["image_id"] for r in fr[a:b]}
        sel_imgs = {r["image_id"] for r in fr[b:n]}
        res["folds"][k] = {"rows": n, "images": len({r["image_id"] for r in fr}),
                           "training_rows": b - a, "selection_rows": n - b,
                           "selection_images": len(sel_imgs),
                           "train_selection_image_overlap": len(train_imgs & sel_imgs),
                           "images_from_other_fold": sum(1 for i in train_imgs | sel_imgs if tr[str(i)] != k)}
    # caption assignment (make_split.py): train2014 image → model 1 − fold; val2014 → fold value; test2015 → id mod 2
    qd_fold_imgs = {k: {r["image_id"] for r in ok if tr[str(r["image_id"])] == k} for k in (0, 1)}
    final_recs = [r for s in ("train", "dev", "test") for r in load(SPLIT / f"{s}.json")]
    seen_violations = 0
    assign = Counter()
    for r in final_recs:
        if r["image_split"] == "train":
            m = 1 - tr[str(r["image_id"])]
        else:
            m = va[str(r["image_id"])]
        assign[(r["image_split"], m)] += 1
        if r["image_id"] in qd_fold_imgs[m]:
            seen_violations += 1
    res["vqa_questions_checked"] = len(final_recs)
    res["questions_per_(split, caption_model)"] = {f"{s}->g{m}": n for (s, m), n in sorted(assign.items())}
    res["questions_captioned_by_a_model_trained_on_that_image"] = seen_violations
    t3 = load(ROOT / "vqa_final_t3/vqa_final/runs/blip_s0/test2015_question_ids.json")
    test_imgs = {q // 1000 for q in t3}
    res["test2015_questions"] = len(t3)
    res["test2015_images"] = len(test_imgs)
    res["test2015_images_in_qd"] = len(test_imgs & (qd_fold_imgs[0] | qd_fold_imgs[1]))
    res["test2015_images_in_train_or_val2014"] = len({str(i) for i in test_imgs} & (set(tr) | set(va)))
    res["test2015_per_caption_model"] = dict(Counter(i % 2 for i in test_imgs))
    return res


# ---------------------------------------------------------------- splits, answers, 10,000 vs 10,001
def splits_and_answers():
    S = {s: load(SPLIT / f"{s}.json") for s in ("train", "dev", "test")}
    img = {s: {r["image_id"] for r in S[s]} for s in S}
    def vocab(recs):
        c = Counter(majority(r["answers"]) for r in recs)
        return sorted(a for a, n in c.items() if n >= 9 and a)
    dev_vocab = vocab(S["train"])
    final_vocab = vocab(S["train"] + S["dev"] + S["test"])
    empty = [r for r in S["test"] if not majority(r["answers"])]
    return {
        "questions": {s: len(v) for s, v in S.items()}, "images": {s: len(v) for s, v in img.items()},
        "image_overlap": {"train-dev": len(img["train"] & img["dev"]), "train-test": len(img["train"] & img["test"]),
                          "dev-test": len(img["dev"] & img["test"])},
        "answers_rebuilt_from_train_only": len(dev_vocab),
        "matches_development_runs": all(load(ROOT / f"{p}_s0/answers.json") == dev_vocab for _, p in RUNS.values()),
        "answers_rebuilt_final_train": len(final_vocab),
        "matches_final_captions_run": load(ROOT / TESTDEV["captions"][0] / "answers.json") == final_vocab,
        "matches_final_baseline_run": load(ROOT / TESTDEV["baseline"][0] / "answers.json") == final_vocab,
        "test_questions_with_empty_majority_answer": [{"question_id": r["question_id"], "question": r["question"],
                                                       "answers": r["answers"]} for r in empty],
    }


# ---------------------------------------------------------------- bootstrap
def per_question(key, recs):
    path = RUNS[key][1]
    seeds = [{p["question_id"]: p["score"] for p in load(ROOT / f"{path}_s{s}/test_predictions.json")} for s in (0, 1)]
    qids = [r["question_id"] for r in recs]
    single = np.array([[sd[q] for q in qids] for sd in seeds])            # (2, N)
    answers = load(ROOT / f"{path}_s0/answers.json")
    prob = np.mean([np.load(ROOT / f"{path}_s{s}/test_scores.npy").astype(np.float32) for s in (0, 1)], axis=0)
    pred = prob.argmax(1)
    ens = np.array([vqa_score(answers[a], r["answers"]) for a, r in zip(pred, recs)])
    return {"seed_mean": single.mean(0), "seed0": single[0], "seed1": single[1], "ensemble": ens}


def boot(diff, clusters, rng):
    """95% percentile interval of mean(diff) × 100, resampling clusters (or questions) with replacement."""
    uniq, inv = np.unique(clusters, return_inverse=True)
    m = len(uniq)
    s = np.bincount(inv, weights=diff, minlength=m)
    n = np.bincount(inv, minlength=m).astype(float)
    out = np.empty(B)
    for b in range(0, B, 500):
        idx = rng.integers(0, m, size=(min(500, B - b), m))
        out[b:b + len(idx)] = s[idx].sum(1) / n[idx].sum(1)
    lo, hi = np.percentile(100 * out, [2.5, 97.5])
    return [round(float(lo), 2), round(float(hi), 2)]


def bootstrap():
    recs = load(SPLIT / "test.json")
    S = {k: per_question(k, recs) for k in RUNS}
    img = np.array([r["image_id"] for r in recs])
    qid = np.array([r["question_id"] for r in recs])
    typ = np.array([r["answer_type"] for r in recs])
    res = {"questions": len(recs), "images": int(len(np.unique(img))), "resamples": B, "models": {}, "contrasts": []}
    for k, (name, _) in RUNS.items():
        res["models"][k] = {"name": name, **{v: round(100 * float(S[k][v].mean()), 2) for v in S[k]}}
    for a, b in (("large", "base"), ("blip_base", "base"), ("large", "blip_base")):
        for view in ("seed_mean", "ensemble"):
            for t in TYPES:
                m = np.ones(len(recs), bool) if t == "overall" else typ == t
                d = (S[a][view] - S[b][view])[m]
                rng = np.random.default_rng(0)
                res["contrasts"].append({"a": a, "b": b, "view": view, "type": t, "n": int(m.sum()),
                                         "images": int(len(np.unique(img[m]))),
                                         "diff": round(100 * float(d.mean()), 2),
                                         "ci_image_clustered": boot(d, img[m], rng),
                                         "ci_question_level": boot(d, qid[m], rng)})
    return res


# ---------------------------------------------------------------- official test-dev
def official():
    res = {}
    for k, runs in TESTDEV.items():
        seeds = [evalai(ROOT / r / "EvalAI-Test-Dev-Result.json")["test-dev"] for r in runs]
        ens = evalai(ROOT / ENSEMBLE[k]["test-dev"])
        res[k] = {"seed0": seeds[0], "seed1": seeds[1],
                  "mean": {t: round(statistics.mean(s[t] for s in seeds), 3) for t in TYPES},
                  "sd": {t: round(statistics.stdev(s[t] for s in seeds), 3) for t in TYPES},
                  "ensemble_test_dev": ens["test-dev"], "ensemble_test_standard": ens["test-standard"]}
    res["gain_single_mean"] = {t: round(res["captions"]["mean"][t] - res["baseline"]["mean"][t], 2) for t in TYPES}
    res["gain_ensemble_test_dev"] = {t: round(res["captions"]["ensemble_test_dev"][t] - res["baseline"]["ensemble_test_dev"][t], 2)
                                     for t in TYPES}
    res["ensemble_minus_single_mean"] = {k: round(res[k]["ensemble_test_dev"]["overall"] - res[k]["mean"]["overall"], 2)
                                         for k in TESTDEV}
    return res


def md(r):
    f, x, s, b, o = r["data_flow"], r["crossfit"], r["splits"], r["bootstrap"], r["official"]
    sr = f["screening_and_repair"]
    L = ["# Manuscript evidence (generated by `CaptionVQA/scripts/paper_evidence.py`)", "",
         "## 1. Data flow of the QD caption corpus (from the `info` block of the dataset file)", "",
         "| Stage | Model / version | Records |", "|---|---|---:|",
         f"| train2014 questions | — | {f['train2014_questions']:,} |",
         f"| first pass: classification, generation, fast validation (blacklist {f['first_pass']['blacklist_not_visual']:,}, "
         f"OCR {f['first_pass']['ocr_excluded']:,}, duplicates {f['first_pass']['duplicates']}, empty {f['first_pass']['dropped_empty']}, "
         f"validation failures {f['first_pass']['validation_failures']}) | `{f['first_pass']['generator_model']}`, "
         f"classifier {f['first_pass']['classifier_prompt']}, generation {f['first_pass']['generation_prompt']} | "
         f"{f['first_pass']['caption_records']:,} |",
         f"| reclassification (parse-failure fix) | `{f['reclassification']['model']}`, {f['reclassification']['prompt']} | "
         f"{f['reclassification']['rows_after']:,} |",
         f"| semantic screening (judge {sr['judge_prompt']}) | `{sr['model']}` | ok {sr['initial']['ok']:,} / WRONG "
         f"{sr['initial']['WRONG']:,} / GRAMMAR_ISSUE_ONLY {sr['initial']['GRAMMAR_ISSUE_ONLY']:,} |",
         f"| regeneration of WRONG ({sr['regen_rounds']} rounds, T = {sr['regen_temperature']}) | `{sr['model']}`, {sr['regen_prompt']} | "
         f"fixed {sr['fixed']:,}, still WRONG {sr['still_wrong']:,} |",
         f"| final verdicts | — | ok {sr['final']['ok']:,} / GRAMMAR_ISSUE_ONLY {sr['final']['GRAMMAR_ISSUE_ONLY']:,} / WRONG "
         f"{sr['final']['WRONG']:,} |",
         f"| **captioner training (verdict == ok)** | `train.py: load_qd_json` | **{f['captioner_training_rows (verdict == ok)']:,}** "
         f"({f['ok_images']:,} images) |", "",
         f"- Remaining WRONG = {f['wrong_share_of_screened_pct']:.2f}% of the screened records (validator verdict, not a measured error rate).",
         f"- {f['ok_rows_with_suspicious_flag']:,} ok rows carry the heuristic `suspicious` flag "
         f"(reasons over all rows: {sr['suspicious_reasons']}); the 7B judge passed them, they are used.",
         f"- Initial WRONG reasons: {sr['initial_wrong_reasons']}.", "",
         "## 2. Cross-fitting proof", "",
         f"- Fold file seed {x['fold_seed']}; train2014 images per fold {x['train2014_images_per_fold']}; val2014 {x['val2014_images_per_fold']}; "
         f"train/val overlap {x['train_val_image_overlap']}.",
         f"- QD images without a fold: {x['qd_images_without_fold']}; QD images in val2014: {x['qd_images_in_val2014']}.", "",
         "| Captioner | QD rows | images | training rows | selection rows (checkpoint) | selection images | train ∩ selection images | images from the other fold |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k, v in x["folds"].items():
        L.append(f"| g{k} | {v['rows']:,} | {v['images']:,} | {v['training_rows']:,} | {v['selection_rows']:,} | "
                 f"{v['selection_images']:,} | {v['train_selection_image_overlap']} | {v['images_from_other_fold']} |")
    L += ["", f"- VQA questions checked (train + dev + test = final training set): {x['vqa_questions_checked']:,}; "
          f"by (image split → caption model): {x['questions_per_(split, caption_model)']}.",
          f"- **Questions captioned by a captioner that was trained on that image: {x['questions_captioned_by_a_model_trained_on_that_image']}.**",
          f"- test2015: {x['test2015_questions']:,} questions on {x['test2015_images']:,} images; images in the QD corpus "
          f"{x['test2015_images_in_qd']}, in train/val2014 {x['test2015_images_in_train_or_val2014']}; per caption model (id mod 2) "
          f"{x['test2015_per_caption_model']}.", "",
          "## 3. Development splits and answer vocabulary", "",
          f"- Questions {s['questions']}, images {s['images']}, image overlap {s['image_overlap']}.",
          f"- Answers (majority answer seen ≥ 9 times) rebuilt from the 643,110 training questions only: "
          f"{s['answers_rebuilt_from_train_only']:,}; identical to all development runs: {s['matches_development_runs']}.",
          f"- Rebuilt from the 658,111 final training questions: {s['answers_rebuilt_final_train']:,}; identical to the final "
          f"caption run: {s['matches_final_captions_run']}, final baseline: {s['matches_final_baseline_run']}.",
          f"- Diagnostic questions whose majority answer is empty after normalisation (no right/wrong caption label): "
          f"{s['test_questions_with_empty_majority_answer']}. Hence 10,000 labelled + 1 = 10,001.", "",
          f"## 4. Image-clustered bootstrap (10k diagnostic set, {b['questions']:,} questions on {b['images']:,} images, {b['resamples']:,} resamples)", "",
          "| Model | seed 0 | seed 1 | seed mean | 2-seed ensemble |", "|---|---:|---:|---:|---:|"]
    for k, v in b["models"].items():
        L.append(f"| {v['name']} | {v['seed0']:.2f} | {v['seed1']:.2f} | {v['seed_mean']:.2f} | {v['ensemble']:.2f} |")
    L += ["", "| Contrast | view | type | n | Δ | 95% CI, image-clustered | 95% CI, question-level |", "|---|---|---|---:|---:|---|---|"]
    for c in b["contrasts"]:
        L.append(f"| {c['a']} − {c['b']} | {c['view']} | {c['type']} | {c['n']:,} | {c['diff']:+.2f} | "
                 f"[{c['ci_image_clustered'][0]:+.2f}, {c['ci_image_clustered'][1]:+.2f}] | "
                 f"[{c['ci_question_level'][0]:+.2f}, {c['ci_question_level'][1]:+.2f}] |")
    L += ["", "## 5. Official test-dev, single seeds and ensembles", "",
          "| Model | seed 0 | seed 1 | mean ± sd | 2-seed ensemble (test-dev) | ensemble (test-standard) |", "|---|---:|---:|---:|---:|---:|"]
    for k in ("captions", "baseline"):
        v = o[k]
        L.append(f"| {k} | {v['seed0']['overall']:.2f} | {v['seed1']['overall']:.2f} | {v['mean']['overall']:.2f} ± "
                 f"{v['sd']['overall']:.2f} | {v['ensemble_test_dev']['overall']:.2f} | {v['ensemble_test_standard']['overall']:.2f} |")
    L += ["", f"- Caption gain, single-seed means: {o['gain_single_mean']}; ensembles (test-dev): {o['gain_ensemble_test_dev']}.",
          f"- Ensemble minus single-seed mean: {o['ensemble_minus_single_mean']}."]
    return "\n".join(L) + "\n"


def main():
    q = load(QD)
    rows = q["annotations"]
    r = {"data_flow": data_flow(q["info"], rows), "crossfit": crossfit(rows), "splits": splits_and_answers(),
         "bootstrap": bootstrap(), "official": official()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "paper_evidence.json").write_text(json.dumps(r, indent=1, default=int), encoding="utf-8")
    (OUT / "paper_evidence.md").write_text(md(r), encoding="utf-8")
    print(md(r))


if __name__ == "__main__":
    main()
