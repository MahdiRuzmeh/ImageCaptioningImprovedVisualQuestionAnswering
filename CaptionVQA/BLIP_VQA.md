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

## Steps

All commands run from `CaptionVQA/` in Git Bash, with the project venv.

### 1. On your PC: split and image features (done once, ~40 min)

```bash
../.venv/Scripts/python.exe make_split.py --config configs/blip.yaml
../.venv/Scripts/python.exe prepare_regions.py --config configs/blip.yaml
```

- `make_split.py` writes `outputs/blip/split/{train,dev,test}.json` and `data/blip_caption_requests.json` (45,953 questions, committed).
- `prepare_regions.py` writes `outputs/blip/features/regions_{train,dev,test}.pt`: BUTD features for 9,690 images.

### 2. On Kaggle: BLIP captions (~20 min on 2× T4)

New notebook: GPU T4 x2, internet on. Inputs:
- `mnhlfuch/vqa-v2` (train2014 + val2014 images);
- **your BLIP training notebook's output** (Add Input → Notebooks), which holds `model_b/b_blip_full/best.pt`.

```bash
!rm -rf /kaggle/working/repo
!git clone -b feat/captionvqa-blip --depth 1 https://github.com/MahdiRuzmeh/ImageCaptioningImprovedVisualQuestionAnswering.git /kaggle/working/repo
%cd /kaggle/working/repo/QuestionDependentImageCaptioner
!bash scripts/kaggle_captions_vqa.sh
```

Download `/kaggle/working/captions_vqa/captions_blip.json` (~3 MB) and put it at
`CaptionVQA/outputs/blip/captions/captions_blip.json`.

### 3. On your PC: train both models (~1 h for all 7 runs)

```bash
bash scripts/run_blip_vqa.sh
```

This runs:
- **baseline** (no caption), seeds 0, 1, 2;
- **+ BLIP captions**, seeds 0, 1, 2;
- **+ oracle captions**, seed 0: the gold answer is put in front of the BLIP caption, which shows the ceiling;
- **the summary**, `outputs/blip/summary.md`.

Or one run at a time:

```bash
../.venv/Scripts/python.exe train_vqa.py --config configs/blip.yaml --captions none --seed 0
../.venv/Scripts/python.exe train_vqa.py --config configs/blip.yaml --captions blip --seed 0
```

Train both models only after the captions are in place: the shared word vocabulary includes the caption
words, so both models start identical.

### Output

| File | Content |
|---|---|
| `outputs/blip/runs/<nocap\|blip\|oracle>_s<seed>/results.json` | dev + test accuracy: overall, by answer type, by question type; gate; test with shuffled captions |
| `…/test_predictions.json` | every test question with its caption, prediction and score |
| `outputs/blip/summary.md` | mean ± std over seeds: overall, yes/no, number, other; per question type, BLIP − baseline |

The **shuffled-caption** test gives each test question the caption of another image's question. If accuracy
drops, the model really uses the caption. The **gate** shows how much it relies on the image (1) vs the caption (0).
