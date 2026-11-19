# Round 2: better fold captioners

Goal: more captions that contain the answer. When the caption contains the answer, the strong VQA model gets +9 points on
open questions, and the oracle ceiling is 73.6. First fold 0 only, two variants. Then compare with the current
fold-0 captioner on the same 10,000 QD rows of fold 1 (images neither captioner trained on).

| | Current fold captioner | Variant A | Variant B |
|---|---|---|---|
| ViT | frozen | **top 4 of 12 layers + post-LayerNorm trained, lr × 0.1** | as A |
| Epochs | 4 | **6** | 6 |
| Loss | label-smoothed CE | as before | **answer words in the caption × 3** |
| Batch | 48 / GPU | 24 / GPU × 2 accumulation (same 96 global) | as A |

Config: `configs/b_blip_fold_v2.yaml` (`unfreeze_vision_layers`, `vision_lr_scale`, `answer_weight`).

**Answer weighting:**
- For a text or number answer, the answer's tokens in the caption count × 3. This includes each content word of a multi-word answer, and digit ↔ word forms (2 ↔ two).
- For "no", the negation token counts × 3 (not / no / none / n't …).
- For "yes", nothing changes.
- The loss is the weighted mean over caption tokens.

## Kaggle (one notebook per account at a time; GPU T4 x2, internet on)

Inputs for both: `mnhlfuch/vqa-v2`, `questiondependentcaptions`.

```bash
!rm -rf /kaggle/working/repo
!git clone -b feat/captionvqa-blip --depth 1 https://github.com/MahdiRuzmeh/ImageCaptioningImprovedVisualQuestionAnswering.git /kaggle/working/repo
%cd /kaggle/working/repo/QuestionDependentImageCaptioner
!VARIANT=A FOLD=0 bash scripts/kaggle_blip_fold_v2.sh
```

Variant B: the same cells with `VARIANT=B`. The run trains, then scores 10,000 rows of fold 1 (beam 3, both GPUs).
It ends with `BLIP_FOLD0_V2A_DONE`. If it stops on the time budget, run again with the notebook's own output attached
(resumes from `last.pt`).

Output `/kaggle/working/model_b/b_blip_fold0_v2<A|B>/`: `fold_eval.json` + `fold_eval.log` (the result),
`train.log`, `best.pt` (keep for re-captioning).

## Reference: current fold captioners (same rows, from `captions_crossfit.json`)

`scripts/eval_fold_captioner.py --captions …`, 10,000 rows, seed 0:

| Captioner (scored on) | AnswerHit | number | other | YesNoPolarity | CIDEr | BLEU-4 |
|---|---:|---:|---:|---:|---:|---:|
| fold 0 (fold-1 rows) | **58.10** | 44.97 | 62.77 | **75.10** | 6.39 | 64.56 |
| fold 1 (fold-0 rows) | 57.89 | 45.64 | 62.26 | 74.93 | 6.38 | 64.20 |

These are higher than the 48% on all VQA questions because the QD rows are the "ok"-verdict questions.
A variant counts as clearly better if AnswerHit improves by ≥ 2 points without a polarity drop. Only then are
both folds retrained and all questions re-captioned.
