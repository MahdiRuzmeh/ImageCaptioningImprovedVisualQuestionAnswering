# Human review of the question-dependent caption dataset

Independent check of the caption quality, separate from the Qwen judges (supervisor comments, item 3). It estimates
the real error rate of the 277,318 `ok` captions that train the captioner. It does **not** change the dataset or
retrain anything.

## Sample (fixed before anyone looks at the captions)

| Group | Population | Sample |
|---|--:|--:|
| `ok`, no "suspicious" flag | 247,474 | 300 |
| `ok`, with "suspicious" flag | 29,844 | 200 |

`make_review_sample.py` draws both with seed 20261128 and shuffles them together. The page `review.html` does not show
the group, the verdict or the flag. `sample_key.json` holds them: **do not open it while reviewing.** The sample is
committed to git before the review, which records that it was fixed in advance.

## What each reviewer judges (Yes / No for each item)

1. The caption states the **reference answer** correctly (the majority answer, or an equivalent form).
2. It keeps the **subject and the relation** of the question.
3. It adds **no fact** that is not in the question or the answer.
4. It is **grammatical** and natural English.
5. Looking at the image, the question is **visually answerable** (no opinion, outside knowledge or reading text).

A caption counts as **correct** when 1, 2 and 3 are all Yes. Grammar (4) and visual answerability (5) are reported
separately: 4 does not make a caption wrong, and 5 checks the question classification of step 1.

Judge the caption against the question and the reference answer, using the image to understand the question.
Don't judge whether the human answer itself is right.

## Who reviews

- **Reviewer A** (one person, e.g. you): all 500 items. About 20–30 s per item, so 3–4 hours in a few sittings.
- **Reviewer B** (a second person, e.g. a colleague or your supervisor): items **1–100** only. These are marked
  "reviewed by both reviewers"; they are a random mix of both groups. B must not see A's labels.
- Afterwards, A and B discuss only the items where they disagree (listed by the analysis) and agree on one label.
  Save that as a third file (same format) and pass it as `--adjudicated`.

One reviewer is enough for the rates. The second reviewer on 100 items shows that the labels are reproducible
(Cohen's kappa), which reviewers of the paper will ask for. Because Reviewer A built the dataset, the blinding
(groups mixed, no verdicts shown) matters.

## Steps

1. Open `review.html` in Chrome or Edge (double-click). The images load from `src/dataset/train2014/`.
2. Enter your name, judge each item (← → keys move between items), and add a comment when unsure.
   Progress is saved in the browser. Export regularly with **Export labels (JSON)**.
3. Reviewer B uses their own computer or browser profile, or the same page after exporting and
   importing an empty file (the browser keeps one set of labels per page).
4. Analyse:
   ```
   python audit/human_review/analyse_review.py --labels review_labels_A.json review_labels_B.json
   python audit/human_review/analyse_review.py --labels review_labels_A.json review_labels_B.json --adjudicated review_labels_final.json
   ```
   Output: `review_results.md` / `.json`, with the rate and 95% Wilson interval per criterion and group, the
   population-weighted rate
   e = (247,474 / 277,318) · e_unflagged + (29,844 / 277,318) · e_flagged (with a stratified 95% interval),
   and agreement / kappa on items 1–100 with the list of items to resolve.
