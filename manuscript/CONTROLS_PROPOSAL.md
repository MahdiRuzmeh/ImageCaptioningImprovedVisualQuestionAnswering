# Proposal: two matched controls (to be approved before running)

The official gain (+3.60) shows that our question-dependent (QD) captions help the no-caption baseline. Two questions remain:
does the **question conditioning** matter, and is the gain more than the **extra parameters** of the caption branch?
Both controls use the development protocol only (643,110 training questions, 5,000 validation, 10,001 development set) and
no test-standard submission.

| | C1: generic captions | C2: null caption (capacity) |
|---|---|---|
| Question answered | do QD captions beat question-independent captions of the same captioner? | is the gain more than the extra parameters? |
| Captioner | BLIP-large fine-tuned on **COCO generic captions**, same 2 image folds, same schedule (4 epochs, lr 3e-5, frozen ViT, batch 48 / GPU), checkpoint by CIDEr on own-fold selection images, beam 3; one caption per image | none |
| VQA | MCAN, identical to the final development model (batch 64, GloVe 840B, caption dropout 0.3, no gate), 2 seeds | same model with the caption branch, but every question gets the same fixed caption ("an image."): the branch and its parameters are active, the caption carries no information (an all-PAD caption would be masked out and test nothing) |
| Compared with | QD BLIP-large 69.76; baseline 66.39 | baseline 66.39 (56.46 M parameters) vs. 58.77 M |
| Reported | overall + answer types, seed means, paired image-clustered bootstrap vs. QD and vs. baseline; shuffled-caption test | same |
| Cost (estimate) | 2 fold captioners ≈ 2 × 4 h + captions ≈ 2 h + VQA 2 seeds ≈ 6 h (≈ 16 GPU h) | VQA 2 seeds ≈ 6 h |
| Expected outcome | between baseline and QD; a clear gap to QD supports question conditioning | ≈ baseline; if it is clearly above the baseline, part of the gain is capacity |

Notes:
- Fairness: generic captions are cross-fitted in the same way (a captioner never captions images of its own fold), so they can
  carry no more image-specific training signal than the QD captions.
- Results go into Table 11 of the manuscript; nothing else in the paper changes.
