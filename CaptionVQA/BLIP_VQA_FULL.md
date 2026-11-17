# Fair control: VQA on all of train2014, baseline vs + cross-fitted BLIP captions

In the first experiment ([BLIP_VQA.md](BLIP_VQA.md)) BLIP captions raised VQA accuracy from 46.6 to 59.8. But the
captioner had been trained on 264k train2014 question–answer pairs, while the VQA models saw only 31k questions.
This experiment gives **both** models the same supervision:

| Model | VQA training data | Captioner training data |
|---|---|---|
| Baseline-full | all train2014: 443,757 questions, 82,783 images | — |
| + cross-fitted BLIP | the same 443,757 questions | the same train2014 QD captions, but never on the image it captions |

Test and dev are **identical** to the first experiment (10,001 / 5,000 val2014 questions).

## Cross-fitting

```
train2014 images ──► fold 0 (41,391 images)   ──► BLIP fold-0 trained on their 138,557 QD captions
                 └─► fold 1 (41,392 images)   ──► BLIP fold-1 trained on their 138,761 QD captions

captions for fold-0 images  ◄── BLIP fold-1 (never saw them)
captions for fold-1 images  ◄── BLIP fold-0 (never saw them)
dev/test (val2014) images   ◄── half from fold-0, half from fold-1 (same caption quality as in training)
```

The assignment is fixed in `data/crossfit_folds.json` (seeded, committed); both the captioners and the VQA
side read it.

## Steps: 4 Kaggle notebooks

Use GPU T4 x2 and internet on for every notebook, and run each as **Save Version → Save & Run All**.

| # | Notebook | Needs | Time |
|---|---|---|---|
| 1 | BLIP fold 0 | — | ~2–2.5 h |
| 2 | BLIP fold 1 | — | ~2–2.5 h |
| 3 | VQA baseline-full | — | ~1.5 h |
| 4 | VQA + cross-fitted BLIP | outputs of 1 and 2 | ~4.5 h |

Notebooks 1, 2 and 3 are independent: run them at the same time (e.g. 1 + 3 on one account, 2 on the other).
Start 4 when 1 and 2 are done.

### Notebooks 1 and 2: cross-fitted BLIP captioners

Inputs: `questiondependentcaptions`, `mnhlfuch/vqa-v2`.

```bash
!rm -rf /kaggle/working/repo
!git clone -b feat/captionvqa-blip --depth 1 https://github.com/MahdiRuzmeh/ImageCaptioningImprovedVisualQuestionAnswering.git /kaggle/working/repo
%cd /kaggle/working/repo/QuestionDependentImageCaptioner
!FOLD=0 bash scripts/kaggle_blip_fold.sh
```

Notebook 2 is the same with `FOLD=1`. Each trains with `torchrun` on both GPUs (same settings as the full BLIP:
frozen ViT, question prompt, 4 epochs, best val CIDEr). Output: `/kaggle/working/model_b/b_blip_fold<k>/best.pt`.
It ends with `BLIP_FOLD0_DONE` / `BLIP_FOLD1_DONE`.

### Notebook 3: VQA baseline-full

Inputs: `mnhlfuch/vqa-v2`, `questiondependentcaptions`, and **the BUTD packed cache from model A** (the dataset
you made from the `STAGES=extract` output; it holds `butd_packed/meta.json`). The cache saves ~3 h:
79,568 images are copied, only ~6k are extracted.

```bash
!rm -rf /kaggle/working/repo
!git clone -b feat/captionvqa-blip --depth 1 https://github.com/MahdiRuzmeh/ImageCaptioningImprovedVisualQuestionAnswering.git /kaggle/working/repo
%cd /kaggle/working/repo/CaptionVQA
!MODE=baseline bash scripts/kaggle_vqa_full.sh
```

Ends with `VQA_FULL_BASELINE_DONE`.

### Notebook 4: VQA + cross-fitted BLIP captions

Inputs: the same three as notebook 3, **plus the outputs of notebooks 1 and 2** (Add Input → Notebooks). If they
ran on your other account, share them first (notebook → Share → add your other account), or save the two
`best.pt` files as a dataset.

```bash
!rm -rf /kaggle/working/repo
!git clone -b feat/captionvqa-blip --depth 1 https://github.com/MahdiRuzmeh/ImageCaptioningImprovedVisualQuestionAnswering.git /kaggle/working/repo
%cd /kaggle/working/repo/CaptionVQA
!MODE=crossfit bash scripts/kaggle_vqa_full.sh
```

- **Captions:** generated with both GPUs at once (fold-0 model on GPU 0, fold-1 model on GPU 1), ~229k each, about 2.8 h.
- **Resume:** captions are saved every 10k. If the session stops, attach this notebook's output to a new version and it continues.
- Ends with `VQA_FULL_CROSSFIT_DONE`.

### How the two GPUs are used

| Step | GPU 0 | GPU 1 |
|---|---|---|
| BLIP fold training | `torchrun` (DDP) | `torchrun` (DDP) |
| BUTD features | shard 0 of the missing images | shard 1 |
| Captions | fold-0 model | fold-1 model |
| VQA training | seed 0, then seed 2 | seed 1 |

VQA training runs whole seeds in parallel instead of DDP: the model has ~6M parameters, so DDP would spend
more time syncing than computing.

## What to bring back

| Notebook | Folder | Put it at |
|---|---|---|
| 3 | `/kaggle/working/vqa_full_baseline/` | `CaptionVQA/vqa_full_baseline/` |
| 4 | `/kaggle/working/vqa_full_crossfit/` | `CaptionVQA/vqa_full_crossfit/` |
| 1, 2 | `train.log` only (keep `best.pt` on Kaggle) | `QuestionDependentImageCaptioner/kaggle_blip_fold<k>/` |

Then: `python summarize_vqa.py --runs vqa_full_baseline vqa_full_crossfit --out vqa_full_summary.md`.
