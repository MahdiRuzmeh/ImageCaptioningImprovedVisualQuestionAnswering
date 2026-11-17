# VQA baseline vs the same model + BLIP question-dependent captions

Does a question-dependent captioner help a VQA model? Two models are trained on the same data with the
same settings. They differ **only** in the caption branch.

## Models (`models_vqa.py`)

**Baseline: Bottom-Up Top-Down VQA** (Anderson et al. 2018)

```
question ──► word embedding (300) ──► GRU (512) ──► q
image ──► BUTD Faster R-CNN (frozen, Visual Genome) 36 regions + 1 mean token × 2048 ──► LN + Linear 512 ──► v_1..v_37
top-down attention:  α_i = softmax(w · (FC(v_i) ⊙ FC(q))),  v̂ = Σ α_i v_i
h = FC(q) ⊙ FC(v̂)  ──► classifier (512 → 1024 → answers), soft-score BCE
```

**+ BLIP captions**: the same model plus one branch:

```
BLIP caption ──► the same word embedding ──► BiGRU (2 × 256) ──► one state per caption word c_1..c_L
question-guided attention over the words:  β_j = softmax(w · (FC(c_j) ⊙ FC(q))),  ĉ = Σ β_j c_j
gate:  g = σ(W [q; v̂; ĉ])        h = FC(q) ⊙ (g ⊙ FC(v̂) + (1 − g) ⊙ FC(ĉ))
```

The caption is read as a **sequence**, not an average. The BiGRU state of "running" in *"the dog is not
running"* carries the "not", and the question picks the words it needs.

## Data (`configs/blip.yaml`, `make_split.py`)

| Split | Source | Questions | Images |
|---|---|---:|---:|
| Train | train2014 images the BLIP captioner was **not** trained on | 30,952 | 6,911 |
| Dev (best epoch) | val2014 | 5,000 | 879 |
| Test (reported) | val2014, other images | 10,001 | 1,900 |

- Dev and test are the same as the earlier CaptionVQA benchmark.
- Answer vocabulary: answers seen at least 3 times in train.
- Word vocabulary: question and caption words seen at least 2 times in train, shared by both models.
- Accuracy: official VQA v2, reported overall, by answer type (yes/no, number, other) and by question type (65 types).

## Steps: two Kaggle notebooks, run at the same time

Each notebook does everything itself:
1. builds the split and checks it is identical to the committed one (`data/blip_caption_requests.json`);
2. extracts BUTD features (train on GPU 0, dev + test on GPU 1);
3. **blip only:** generates the BLIP captions;
4. trains 3 seeds and tests each;
5. writes `summary.md`.

### Notebook 1: baseline (no captions, ~45 min)

GPU T4 x2, internet on. Inputs: `mnhlfuch/vqa-v2`, `questiondependentcaptions`.

```bash
!rm -rf /kaggle/working/repo
!git clone -b feat/captionvqa-blip --depth 1 https://github.com/MahdiRuzmeh/ImageCaptioningImprovedVisualQuestionAnswering.git /kaggle/working/repo
%cd /kaggle/working/repo/CaptionVQA
!MODE=baseline bash scripts/kaggle_vqa.sh
```

### Notebook 2: + BLIP captions (~75 min)

GPU T4 x2, internet on. Inputs: `mnhlfuch/vqa-v2`, `questiondependentcaptions`, and **your BLIP training notebook's
output** (Add Input → Notebooks, it holds `model_b/b_blip_full/best.pt`).

```bash
!rm -rf /kaggle/working/repo
!git clone -b feat/captionvqa-blip --depth 1 https://github.com/MahdiRuzmeh/ImageCaptioningImprovedVisualQuestionAnswering.git /kaggle/working/repo
%cd /kaggle/working/repo/CaptionVQA
!MODE=blip bash scripts/kaggle_vqa.sh
```

Run both as **Save Version → Save & Run All**. Each prints `VQA_BASELINE_DONE` / `VQA_BLIP_DONE` at the end.

### What to bring back

| Notebook | Folder | Put it at |
|---|---|---|
| 1 | `/kaggle/working/vqa_baseline/` | `CaptionVQA/outputs/vqa_baseline/` |
| 2 | `/kaggle/working/vqa_blip/` | `CaptionVQA/outputs/vqa_blip/` |

Each folder holds `runs/`, `summary.md`, the split, and (notebook 2) `captions/captions_blip.json`. The region
features are deleted at the end to keep the output small. The `best.pt` files (~25 MB each) are optional.
With both folders, I combine them into one table: baseline vs + BLIP, with mean ± std over 3 seeds.

The two models start from the same word table for the question words. Notebook 2 appends the caption words after
them, and the baseline never uses those rows, so the notebooks don't depend on each other.

### Running on your PC instead (optional)

```bash
../.venv/Scripts/python.exe make_split.py --config configs/blip.yaml
../.venv/Scripts/python.exe prepare_regions.py --config configs/blip.yaml
bash scripts/run_blip_vqa.sh          # needs outputs/blip/captions/captions_blip.json
```

### Output

| File | Content |
|---|---|
| `outputs/blip/runs/<nocap\|blip\|oracle>_s<seed>/results.json` | dev + test accuracy: overall, by answer type, by question type; gate; test with shuffled captions |
| `…/test_predictions.json` | every test question with its caption, prediction and score |
| `outputs/blip/summary.md` | mean ± std over seeds: overall, yes/no, number, other; per question type, BLIP − baseline |

The **shuffled-caption** test gives each test question the caption of another image's question. If accuracy
drops, the model really uses the caption. The **gate** shows how much it relies on the image (1) vs the caption (0).

## Results (2026-11-17, two Kaggle notebooks, test = 10,001 val2014 questions)

Full table, including all 65 question types: [vqa_summary.md](vqa_summary.md) (`python summarize_vqa.py --runs vqa_baseline vqa_blip`).

| Model | Seeds | Overall | Yes/no | Number | Other | Shuffled captions | Gate (1 = image) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline (no caption) | 3 | **46.56 ± 0.44** | 67.80 ± 0.40 | 28.31 ± 0.50 | 35.53 ± 0.55 | — | — |
| + BLIP captions | 3 | **59.79 ± 0.50** | 78.49 ± 0.82 | 45.03 ± 1.82 | 49.73 ± 1.01 | 32.30 ± 0.04 | 0.55 |
| + oracle captions (ceiling) | 1 | **83.30** | 100.00 | 86.44 | 69.82 | 18.60 | 0.47 |

**Gain of BLIP captions:** +13.2 overall: yes/no +10.7, number +16.7, other +14.2. The largest gains are on colors
("what color is the" +28.9) and counting ("how many" +19.6).

### Checks

- **The model reads the captions.** With another image's caption, accuracy falls to 32.3%, far below the baseline.
  The gate (0.55) uses image and caption about equally.
- **No image leakage.** All test questions are on val2014 images; none is in the captioner's data.
- **Caption quality on the VQA test:** the answer is in the BLIP caption for 49.6% of the non-yes/no questions, and
  the polarity is right for 69.6% of the yes/no questions. This is lower than on the QD test (59% / 76%), because VQA
  also contains questions that aren't directly visual.
- **Where the gain comes from** (seed 0):

| Test questions | n | Baseline | + BLIP | Δ |
|---|---:|---:|---:|---:|
| non-yes/no, caption contains the answer | 3,111 | 48.7 | 79.7 | **+31.0** |
| non-yes/no, caption doesn't | 3,158 | 19.6 | 17.7 | −1.9 |
| yes/no, caption polarity right | 2,595 | 67.0 | 90.5 | **+23.5** |
| yes/no, caption polarity wrong | 1,136 | 69.4 | 53.5 | **−15.9** |

  Correct captions help a lot. Wrong yes/no captions hurt: the model trusts them too much.

### Caveat: amount of supervision

The BLIP captioner was trained on 264k train2014 question–answer pairs (as captions). The VQA models only see 30,952
questions. Part of the gain may therefore be knowledge transferred from the captioner's larger training data, not
only from the captions as a representation. A fair control is a baseline trained on all of train2014 (~443k
questions, no captions needed), compared with a caption model whose captions come from cross-fitted captioners.
