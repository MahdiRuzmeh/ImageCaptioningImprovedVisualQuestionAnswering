# Proposal: two caption controls (to be approved before running)

The official gain (+3.60 test-standard, +3.59 test-dev) shows that our question-dependent (QD) captions help the
no-caption baseline. Two questions remain: does the **question conditioning** matter, and is the gain more than the
**extra parameters** of the caption branch? Both controls repeat the final protocol exactly and are scored on the
official **test-dev** only (no test-standard submission, no tuning).

| | C1: generic captions | C2: fixed caption (capacity) |
|---|---|---|
| Question answered | do QD captions beat generic captions from the same starting model? | is the gain more than the extra parameters? |
| Captions | `Salesforce/blip-image-captioning-large` **as released** (COCO-caption fine-tuned, no training of ours), no question, prefix "a picture of", beam 3; one caption per image, given to all its questions; train2014 + val2014 + test2015 | the same fixed text ("an image.") for every question |
| VQA | final model as T3: MCAN, caption dropout 0.3, no gate, all 658,111 train2014 + val2014 questions, 10 epochs, seeds 0 and 1, answers all test2015 | same |
| Compared with (test-dev) | QD captions: singles 72.13 / 72.13, ensemble 72.87; baseline: 68.30 / 68.31, 69.28 | baseline (56.46 M parameters vs 58.77 M) |
| Cost | captions ≈ 1.5 h (123,287 + 81,434 images, 2 × T4) + VQA ≈ 4 h | VQA ≈ 4 h |
| Reading | close to 69.3 → captions help only when question-dependent; clearly below 72.9 → question conditioning is the gain; close to 72.9 → the claim must be weakened | ≈ baseline expected; clearly above → part of the gain is capacity |

Why the released BLIP-Large is the right generic captioner: our QD captioner starts from the same checkpoint, so the
two caption sets differ only by our QD fine-tuning and the question prompt. The checkpoint's COCO fine-tuning (Karpathy
split) covers the train2014 and most val2014 images but no test2015 image; both captioners share this exposure. Generic
captions are not derived from VQA answers, so cross-fitting is not needed.

Results go into Table 11 of the manuscript.

## Notebook steps (C1)

| Notebook | Settings / inputs | Command (cell) | Output |
|---|---|---|---|
| G1: generic captions, train + val | GPU T4 × 2, internet on; inputs: `mnhlfuch/vqa-v2` | `PART=trainval bash scripts/kaggle_generic_captions.sh` | `generic_captions/captions_crossfit_generic.json` |
| G2: generic captions, test2015 | same + the dataset with test2015 images and `v2_OpenEnded_mscoco_test2015_questions.json` (as in T2) | `PART=test2015 bash scripts/kaggle_generic_captions.sh` | `generic_captions/captions_test2015_generic.json` |
| G3: final VQA + generic captions | inputs as T3, but attach G1 and G2 outputs instead of the QD caption outputs | `MODE=crossfit FINAL=1 SEEDS="0 1" CONFIG=configs/blip_final.yaml CAPTION_DROPOUT=0.3 CAPTIONS_FILE=captions_crossfit_generic.json CAPTIONS_TEST_FILE=captions_test2015_generic.json RUN=vqa_final_generic bash scripts/kaggle_vqa_strong.sh` | `vqa_final_generic_blip_s0/s1_test2015.json`, `vqa_final_generic_ensemble_test2015.json` → EvalAI test-dev |

Each notebook starts with
`!git clone -b master --depth 1 https://github.com/MahdiRuzmeh/ImageCaptioningImprovedVisualQuestionAnswering.git /kaggle/working/repo`
and `%cd /kaggle/working/repo/CaptionVQA`, then `!<command>`. G1 and G2 can run at the same time on two accounts.
