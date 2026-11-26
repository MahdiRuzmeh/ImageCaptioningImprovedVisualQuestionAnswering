# Verification note for manuscript v1

This note answers the supervisor's author checklist (draft "Revised V2") against the code and the archived files. All numbers
come from `CaptionVQA/scripts/paper_evidence.py` (output: `evidence/paper_evidence.md` / `.json`) unless another file is named.

## 1. Ensemble, answer vocabulary, official results

| Checklist item | Finding | Evidence |
|---|---|---|
| Ensemble domain (logits, probabilities or scores?) | **sigmoid probabilities**, averaged over seeds 0 and 1; answer = argmax of the mean | `CaptionVQA/scripts/test2015_results.py`, `CaptionVQA/ensemble_vqa.py` |
| Same answer vocabulary in both final models | yes: 3,148 answers, identical lists | `vqa_final_t3/.../blip_s0/answers.json` = `vqa_final_t4/.../nocap_s0/answers.json`; rebuilt from the training questions only |
| Answer vocabulary only from training data | the majority answer of each training question, kept if it occurs ≥ 9 times: 3,091 (development, 643,110 questions), 3,148 (final, 658,111 questions); both rebuilt lists equal the lists in the runs | `train_vqa.py: Vocab`, `paper_evidence.py` §3 |
| Official files | prediction files and EvalAI result JSONs kept next to each other | `CaptionVQA/vqa_final_t3/vqa_final/`, `CaptionVQA/vqa_final_t4/vqa_final_baseline/` |
| Submission dates | captions ensemble 24 Nov 2026 21:25; baseline ensemble 25 Nov 2026 06:23 (IDs are archived locally and not printed in the paper) | EvalAI |
| Single seeds | test-dev: captions 72.13 / 72.13, baseline 68.30 / 68.31; gain of the means +3.82 | `runs/*/EvalAI-Test-Dev-Result.json` |

## 2. Cross-fitting

| Check | Result |
|---|---|
| Fold file | `CaptionVQA/data/crossfit_folds.json`, seed 0; train2014 41,391 / 41,392 images; val2014 20,251 / 20,253; no image in both |
| Corpus rows per captioner | g0: 138,557 rows on 39,750 images; g1: 138,761 on 39,818 |
| Checkpoint-selection data | the last ~1,500 rows of the captioner's **own** fold, cut at an image boundary (`make_fold_qd.py`, `snap_row_range`): g0 1,488 rows / 411 images, g1 1,500 / 429; 0 images shared with the training rows; 0 images of the other fold |
| Caption assignment | train2014 image → captioner `1 − fold`; val2014 image → its fold value; test2015 image → `image_id mod 2` (`make_split.py`, `scripts/kaggle_caption_beams.sh`) |
| Leakage test | of 658,111 train2014 + val2014 questions, **0** are captioned by a captioner whose training rows contain that image |
| test2015 | 447,793 questions on 81,434 images; 0 of them in the caption corpus or in train/val2014 |
| Answers at inference | the caption prompt is only `"question: {q} caption:"` + the image (`scripts/blip_caption_vqa.py`) |
| Development splits | train 643,110 q / 120,508 images; validation 5,000 / 879; development set 10,001 / 1,900; no shared images |

## 3. Corpus counts, Qwen models, validator

All counts are stored in the `info` block of `dataset/vqa_v2_question_dependent_captions_train2014.json` (the file used to train the
captioners) and add up exactly:

```
443,757 train2014 questions
 − 21,272 not visual (blacklist) − 15,824 OCR − 88 duplicates − 8 empty − 250 failed validation = 406,315   (qwen2.5:3b-instruct-q4_K_M)
 − 122,608 re-classified as not directly visual (opinion 48,468, knowledge 62,631, OCR 11,509) = 283,707  (3B, prompt v20)
 judge (qwen2.5:7b-instruct-q4_K_M, j2): ok 250,987 / WRONG 28,153 / GRAMMAR_ISSUE_ONLY 4,567
 regenerate WRONG (r2 + generation v10, T = 0.3, ≤ 2 rounds): fixed 26,476, still WRONG 1,677
 final: ok 277,318 / GRAMMAR_ISSUE_ONLY 4,712 / WRONG 1,677   (250,987 + 26,476 − 145 = 277,318)
```

- **Which rows train the captioner:** `QuestionDependentImageCaptioner/train.py: load_qd_json` keeps only
  `caption_quality_verdict == "ok"` → **277,318 rows**. The 4,712 GRAMMAR_ISSUE_ONLY and 1,677 WRONG rows are **not** used.
  (Note for the authors: the 7B judge puts captions whose only problem is grammar into GRAMMAR_ISSUE_ONLY, which is a separate
  verdict from `ok`. If grammar-issue captions should be used, the loader must accept that verdict and the corpus grows to 282,030.)
- **Suspicious flag:** 30,585 rows carry a heuristic flag (extra words 29,970, polarity flip 778); 29,844 of them are `ok` and are used.
- **Prompts and rules:** classifier prompt v20, generation prompt v10, fast validator v11, LLM validator v10
  (`QuestionDependentCaptionGenerator/`: `question_classifier.py`, `llm_prompts.py`, `validation/`); the fast validator rules are
  3–30 words, word-overlap fail < 0.3 / pass ≥ 0.5, relation ratio ≥ 0.5. The 7B judge (j2) and regeneration (r2)
  prompts and code are in `QuestionDependentCaptionGenerator/caption_errors/` (`judge.py`, `regenerate.py`, `suspicious.py`,
  `judge_captions.py`), run on Kaggle with `kaggle/kaggle_judge_runner.py`; calibration set:
  `audit/DevAuditor/caption_error_audit_manual.json` (298 hand-labelled captions).

## 4. Mathematical description vs. code

| Item | Code |
|---|---|
| Caption loss | Hugging Face BLIP loss: label-smoothed cross-entropy (ε = 0.1), mean over caption tokens; prompt and padding labels are −100 (`train_blip.py`) |
| Beam score | `generate(num_beams=3)`, length penalty 1 → average token log-probability; `--with-scores` saves it (`scripts/blip_caption_vqa.py`) |
| Checkpoint criterion | `select_metric: answer` = (AnswerHit · n_open + polarity · n_yes/no) / (n_open + n_yes/no) |
| Decoder layer | `x = LN(x + SA(x)); x = LN(x + GA(x, q)); x = LN(x + FFN(x))` (`models_vqa.py`) |
| Type embeddings | regions and caption tokens each get one (`type_emb`) |
| Output | `classifier(LN(AttFlat(q) + AttFlat(x)))`, AttFlat output 1,024 |
| Caption dropout | the whole caption of a question is replaced by PAD with probability 0.3 (training only) |
| VQA loss (BCE) | binary cross-entropy with soft targets, **summed over answers, averaged over the batch** (Eq. 16) |
| Soft target | `soft_targets` = the VQA accuracy formula of each distinct answer (`common.py`) |
| Answer normalisation | official evaluator rules (`common.normalize_answer`) |
| AnswerHit / polarity | `metrics.py: answer_in_caption`, `caption_negated` (negation word list in the paper) |

**What "BCE" means.** Binary cross-entropy: each of the 3,148 answers is treated as its own yes/no decision with a sigmoid score
and a soft target between 0 and 1 (how many annotators gave that answer). The loss adds this error over all answers and then
averages over the questions in the batch. The draft's Eq. (16) had no batch average; v1 adds it.

## 5. Statistics

- **10,000 vs 10,001:** question 475660002 ("What letter is in red on the communist sign?", all answers "a") normalises to an empty
  answer (articles are removed), so its caption has no right/wrong label. It still counts in VQA accuracy.
- **Bootstrap:** 5,000 resamples of the 1,900 images of the development set (questions of an image stay together). The image-clustered
  intervals are wider than question-level intervals by at most 0.10 points overall (up to 0.35 for the 1,359 number questions). Main results: BLIP-Large captions − no captions +3.37
  [+2.83, +3.92]; BLIP-Large − BLIP-base +1.45 [+0.94, +1.96]; ensembles +3.00 [+2.35, +3.65].

## 6. Wording corrected in v1

- the 10k set is called the **development set** (repeatedly used), not a test set;
- the development → official shift (≈ +2.5 for both models) is attributed to 15,001 extra training questions, 3,148 vs 3,091 answers
  and a different question sample, not to the captions;
- Sharma & Jalal values: from Table 6 of the original article (confirmed by the author);
- "MCAN, GloVe 6B" row: batch 128; the later MCAN rows batch 64.
