# Round 2: fold captioners with part of the ViT trained

Goal: more captions that contain the answer. The fold-0 captioner was retrained and compared with the current one on the
same 10,000 QD rows of fold 1 (images neither captioner trained on). Full results and discussion:
`../CaptionVQA/VQA_ROUND2_REPORT.md`, section 3.

| | Current fold captioner | Round 2 (variant A) |
|---|---|---|
| ViT | frozen | **top 4 of 12 layers + post-LayerNorm trained, lr × 0.1** |
| Epochs | 4 | **6** |
| Batch | 48 / GPU | 24 / GPU × 2 accumulation (same 96 global) |

Config: `configs/b_blip_fold_v2.yaml` (`unfreeze_vision_layers`, `vision_lr_scale`).

## Result

| Fold-0 captioner (10,000 fold-1 rows, beam 3) | AnswerHit | number | other | Polarity |
|---|---:|---:|---:|---:|
| current | 58.10 | 44.97 | 62.77 | 75.10 |
| variant A (this config) | 58.51 | 47.02 | 62.59 | 76.04 |
| variant B (A + answer words × 3 in the loss, **removed**) | 58.36 | 46.11 | 62.71 | 72.69 |

- **Variant A is not clearly better** (+0.4 AnswerHit, not significant). The current captions are kept.
- **Variant B was removed from the code.** Weighting the negation word for "no" answers made 69% of its yes/no captions negative (50% of the answers are "no").

## Kaggle (GPU T4 x2, internet on; ~4.5 h)

Inputs: `mnhlfuch/vqa-v2`, `questiondependentcaptions`.

```bash
!rm -rf /kaggle/working/repo
!git clone -b feat/captionvqa-blip --depth 1 https://github.com/MahdiRuzmeh/ImageCaptioningImprovedVisualQuestionAnswering.git /kaggle/working/repo
%cd /kaggle/working/repo/QuestionDependentImageCaptioner
!FOLD=0 bash scripts/kaggle_blip_fold_v2.sh
```

The run trains, then scores 10,000 rows of the other fold (beam 3, both GPUs) and ends with `BLIP_FOLD0_V2_DONE`.
Output: `/kaggle/working/model_b/b_blip_fold0_v2/` (`fold_eval.json`, `fold_eval.log`, `train.log`, `best.pt`).
If it stops on the time budget, run again with the notebook's own output attached (resumes from `last.pt`).

The current captioners' reference numbers come from the existing captions, without a GPU:

```bash
python scripts/eval_fold_captioner.py --qd ../dataset/vqa_v2_qd_captions_full_train2014.json \
    --folds ../CaptionVQA/data/crossfit_folds.json --fold 0 \
    --captions ../CaptionVQA/vqa_strong_crossfit/captions/captions_crossfit.json --out outputs/fold_eval/fold0_current.json
```
