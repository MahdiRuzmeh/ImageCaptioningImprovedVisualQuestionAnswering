# Reproducing the final results

Final result: VQA v2 **test-standard 73.16** (MCAN + question-dependent BLIP-large captions, caption dropout 0.3, 2-seed
ensemble) vs **69.56** (same MCAN without captions). Paper source: [`manuscript/`](manuscript/).

All GPU steps ran on Kaggle with 2 × T4 (one notebook per step). Paths in the scripts are Kaggle paths; the notebook inputs are
listed per step.

## 0. Data

| File | Source |
|---|---|
| VQA v2 questions + annotations (train2014, val2014), test2015 questions | visualqa.org |
| COCO train2014 / val2014 / test2015 images | cocodataset.org |
| BUTD region features (36 regions, Visual Genome Faster R-CNN) | precomputed caches (`CaptionVQA/scripts/kaggle_val2014_cache.sh`, `kaggle_test2015_cache.sh`) |
| GloVe 840B 300-d | nlp.stanford.edu |
| QD caption corpus `vqa_v2_question_dependent_captions_train2014.json` (277,318 `ok` rows; provenance in its `info` block) | built by `QuestionDependentCaptionGenerator/` |

## 1. QD caption corpus (local, Ollama)

`QuestionDependentCaptionGenerator/` (details: its `README.md`, `kaggle/README.md`, `caption_errors/README.md`):

| Step | Code | Kaggle (2 × T4, one Ollama server per GPU) |
|---|---|---|
| 1–3: classification, generation, validation (`qwen2.5:3b-instruct-q4_K_M`) | `generate.py`, `question_classifier.py`, `caption_rules.py`, `llm_prompts.py`, `validation/` | `python -u /tmp/repo/QuestionDependentCaptionGenerator/kaggle/kaggle_runner.py` |
| question classification re-run (same model) | `reclassify_captions.py` | `kaggle/kaggle_reclassify_runner.py` |
| 4: revalidation and repair (`qwen2.5:7b-instruct-q4_K_M`, judge j2, regeneration r2) | `caption_errors/` | `kaggle/kaggle_judge_runner.py` (calibrate first, then `run.mode = "full"`) |

The runners clone this repository and check out `master`. Counts: `manuscript/evidence/paper_evidence.md` §1.

## 2. Cross-fitted BLIP-large captioners (2 notebooks)

```
cd QuestionDependentImageCaptioner
CONFIG=configs/b_blip_fold_large.yaml NAME=large FOLD=0 bash scripts/kaggle_blip_fold_v2.sh
CONFIG=configs/b_blip_fold_large.yaml NAME=large FOLD=1 bash scripts/kaggle_blip_fold_v2.sh
```

Folds: `CaptionVQA/data/crossfit_folds.json` (seed 0). Each fold model trains only on its own images (`scripts/make_fold_qd.py`)
and selects `best.pt` by AnswerHit + polarity on its own last ~1,500 rows (`select_metric: answer`).

## 3. Captions for all questions

```
cd CaptionVQA
CAPTIONER=large PART=trainval bash scripts/kaggle_caption_beams.sh   # 658,111 train+val questions, other-fold model
CAPTIONER=large PART=test2015 bash scripts/kaggle_caption_beams.sh   # 447,793 test2015 questions, model image_id mod 2
```

## 4. Final VQA models and EvalAI files

```
cd CaptionVQA
# T3: captions, seeds 0 and 1
MODE=crossfit FINAL=1 CONFIG=configs/blip_final.yaml CAPTIONS_FILE=captions_crossfit_large.json \
  CAPTIONS_TEST_FILE=captions_test2015_large.json RUN=vqa_final bash scripts/kaggle_vqa_strong.sh
# T4: baseline, seeds 0 and 1
MODE=baseline FINAL=1 CONFIG=configs/blip_final.yaml RUN=vqa_final_baseline bash scripts/kaggle_vqa_strong.sh
```

Config `configs/blip_final.yaml` (MCAN-small, batch 64, GloVe 840B, 10 fixed epochs on all 658,111 questions). The runner writes
`*_s0_test2015.json`, `*_s1_test2015.json` and `*_ensemble_test2015.json` (mean sigmoid score of the seeds,
`scripts/test2015_results.py`) for upload to EvalAI.

## 5. Development experiments and analysis (local, no GPU)

| Output | Command |
|---|---|
| Manuscript evidence (data flow, cross-fitting proof, bootstrap, official seeds) | `python CaptionVQA/scripts/paper_evidence.py` |
| Manuscript figures | `python manuscript/figures/make_figures.py` |
| Manuscript .docx | `python manuscript/build.py` |
| Earlier reports | `CaptionVQA/scripts/*_analysis.py`, `*_figures.py`; reports `CaptionVQA/VQA_*_REPORT.md` |

## Archived results

| | Path |
|---|---|
| Official ensemble results (test-dev + test-standard) | `CaptionVQA/vqa_final_t3/vqa_final/Ensemble-VQA-With-Large-Blip-Test-Standard-Result-EvalAI.json`, `CaptionVQA/vqa_final_t4/vqa_final_baseline/Ensemble-VQA-BaseLine-Test-Standard-Result-EvalAI.json` |
| Single-seed test-dev results | `CaptionVQA/vqa_final_t3/vqa_final/runs/blip_s{0,1}/EvalAI-Test-Dev-Result.json`, `CaptionVQA/vqa_final_t4/vqa_final_baseline/runs/nocap_s{0,1}/EvalAI-Test-Dev-Result.json` |
| Submitted prediction files, run outputs | local / Kaggle notebook outputs (too large for git) |
