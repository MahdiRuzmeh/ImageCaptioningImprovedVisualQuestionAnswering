# Architecture — SimpleImageCaptioner & SimpleVQA

<!--
  In file architecture-e do model ro ba Finglish tozih mide.
  Marja: Sharma & Jalal (2021) — Image captioning improved visual question answering.
  Stage 1 = captioner | Stage 2 = VQA ba captioner-e load shode.
-->

## Kholase (Overview)

<!--
  Do marhale darim: aval caption ro yad migirim, bad VQA ba captioner.
  Stage 1 roye MSCOCO caption train mishe (bedoon soal).
  Stage 2 roye VQA v2 train mishe va captioner ro load mikone.
-->

| Marhale | Project | Data | Khoroji |
|---------|---------|------|---------|
| **1** | `SimpleImageCaptioner/` | MSCOCO captions | Caption baraye tasvir |
| **2** | `SimpleVQA/` | VQA v2 (soal + javab) | Javab baraye (tasvir, soal) |

```mermaid
flowchart LR
    subgraph S1["Stage 1 — SimpleImageCaptioner"]
        I1[Image] --> RCNN1[Faster R-CNN]
        RCNN1 --> GNN1[RelationGNN]
        GNN1 --> DEC[LSTM Decoder]
        DEC --> CAP[Caption tokens]
    end

    subgraph S2["Stage 2 — SimpleVQA"]
        I2[Image] --> RES[ResNet-101 global]
        I2 --> RCNN2[Faster R-CNN regions]
        Q[Question] --> QGRU[Question GRU]
        RCNN2 --> GNN2[RelationGNN]
        GNN2 --> VATT[v_att]
        QGRU --> VATT
        I2 --> CAPM[Captioner + q_emb]
        Q --> CAPM
        CAPM --> VCAP[v_cap]
        VATT --> FUSE[Fusion]
        VCAP --> FUSE
        FUSE --> LSTM[Dual LSTM]
        QGRU --> LSTM
        RES --> LSTM
        LSTM --> ANS[Answer tokens]
    end

    S1 -->|best.pt| S2
```

---

## Stage 1 — SimpleImageCaptioner

<!--
  Model asli: SimpleImageCaptioner dar captioner_v1.py
  Train: SimpleImageCaptioner/train.py
  Har step decode ba attention roye 32 region anjam mishe.
-->

### Hadaf

Toye marhale 1 faghat **image → caption** yad migirim. Soal vared nemishe (`question_ids=None`).

### Pipeline

```mermaid
flowchart TB
    IMG["Image (3×448×448)"]
    FRCNN["Faster R-CNN (frozen)\n→ 32 regions × 1024"]
    ROI["roi_to_region (trainable)\n→ 32 × 2048"]
    GNN["RelationGNN (optional)\nregions + gnn_delta"]
    INIT["LSTM init\nh,c = tanh(W · mean(regions))"]
    LOOP["Decode loop t=1..T"]

    IMG --> FRCNN --> ROI --> GNN --> INIT --> LOOP

    subgraph LOOP_DETAIL["Har step t"]
        QCTX["qctx = 0 (stage 1)"]
        ATT["attention(regions, h + qctx)\n→ context 512-D"]
        WE["word_emb(token_{t-1})"]
        LSTM["LSTMCell([word; context])"]
        OUT["classifier(h + ctx + word)\n→ logits"]
        QCTX --> ATT
        ATT --> LSTM
        WE --> LSTM
        LSTM --> OUT
    end

    LOOP --> LOOP_DETAIL
```

### Component-ha

| Layer | File | Trainable? | Voroudi → Khorouji |
|-------|------|------------|---------------------|
| `RegionEncoder` | `captioner_v1.py` | `roi_to_region` only | Image → `(N, 32, 2048)` |
| `RelationGNN` | `relation_gnn.py` | Yes | `(N,32,2048)` → `(N,32,2048)` residual |
| `RegionAttention` | `captioner_v1.py` | Yes | regions + `h_{t-1}` → context 512 |
| `word_emb` | `captioner_v1.py` | Yes | token id → 512 |
| `LSTMCell` | `captioner_v1.py` | Yes | `[word; context]` → `h_t` 512 |
| `classifier` | `captioner_v1.py` | Yes | hidden → vocab logits |

<!--
  RegionAttention: h va har region ro be 512 project mikone,
  softmax roye 32 region, weighted sum → 2048, bad ctx_proj → 512.
-->

### Decode (inference)

| Mode | Function | Tozih |
|------|----------|-------|
| Greedy | `_decode_caption()` | Har step argmax |
| Beam | `generate_caption()` | beam=5, length-norm, trigram block |

### Abaad (Paper-aligned defaults)

| Symbol | Size | Tozih |
|--------|------|-------|
| K (regions) | 32 | ROI az Faster R-CNN |
| L (region dim) | 2048 | `v_i ∈ ℝ^2048` |
| LSTM hidden | 512 | `h_t`, `m_t` |
| word_dim | 512 | embedding kalamat caption |
| embed_dim | 512 | working space attention |
| max_caption_len | 20 | + BOS/EOS |

### Train (Stage 1)

```
Input:  (image, caption_ids GT)
Loss:   CrossEntropy roye caption_ids[:, 1:]
Mode:   Teacher forcing (+ optional scheduled sampling)
Output: outputs/<run>/best.pt  (model + vocab)
```

**Special tokens:** PAD=0, BOS=1, EOS=2

---

## Stage 2 — SimpleVQA (VQAModel)

<!--
  VQAModel dar SimpleVQA/train.py hast.
  Captioner az Stage 1 load mishe; faghat q_emb/q_proj trainable.
  Do LSTM baraye javab: lstm_att + lstm_ans.
-->

### Hadaf

Baraye har `(image, question)` yek **javab** tolid kon. Captioner komak mikone ta `v_cap` besazim.

### Pipeline — do khat visual

```mermaid
flowchart TB
    subgraph LINE1["Khat 1 — v_att (question-attended visual)"]
        I1[Image]
        RES["ResNet-101 (frozen)\n→ g: 512"]
        LOC["Faster R-CNN (frozen)\n→ local: 512 × 32"]
        GNN2[RelationGNN]
        Q1["Question → q_emb → GRU → q_vec: 512"]
        ATT2["_attend(regions, q_vec)\n→ v_att: 512"]
        I1 --> LOC --> GNN2 --> ATT2
        Q1 --> ATT2
        I1 --> RES
    end

    subgraph LINE2["Khat 2 — v_cap (caption representation)"]
        I2[Image]
        Q2[Question]
        CAP["Captioner (frozen + q_emb trainable)\nquestion-guided decode"]
        VCAP["v_cap: 512"]
        I2 --> CAP
        Q2 --> CAP
        CAP --> VCAP
    end

    FUSE["Fusion: mul | add | concat"]
    V["v (fused feature)"]
    LINE1 --> FUSE
    LINE2 --> FUSE
    FUSE --> V

    subgraph ANS_DEC["Answer decoder (Dual LSTM)"]
        RES2[g global 512]
        Q3[q_vec 512]
        LA["lstm_att(a_{t-1}, g, h2_{t-1})"]
        LB["lstm_ans(h1_t, h2_{t-1}, v, q_vec)"]
        CLS["Linear → answer logits"]
        LA --> LB --> CLS
        RES2 --> LA
        Q3 --> LB
        V --> LB
    end
```

### Component-ha — VQAModel

| Module | Trainable? | Voroudi → Khorouji |
|--------|------------|---------------------|
| `resnet` + `g_proj` | g_proj only | Image → `g` (512) |
| `detector` + `local_proj` | local_proj only | Image → `(N,32,512)` |
| `q_emb`, `q_gru`, `q_proj` | Yes | question ids → `q_vec` (512) |
| `gnn` (RelationGNN) | Yes | regions → updated regions |
| `_attend` | Yes | regions + q_vec → `v_att` (512) |
| `captioner` | q_emb (+ q_proj) | image + q → caption → `v_cap` |
| `lstm_att`, `lstm_ans`, `out` | Yes | → answer logits |

<!--
  Mohem: VQA do ta q_emb joda dare!
  1) VQAModel.q_emb → baraye v_att va answer LSTM
  2) captioner.q_emb → baraye caption question-guided
  Ina vocabulary joda va train joda hastan.
-->

### Caption integration

Captioner az `SimpleImageCaptioner/outputs/.../best.pt` load mishe:

| Captioner part | Stage 2 status |
|----------------|----------------|
| `word_emb`, LSTM, attention, classifier, GNN | **Frozen** |
| `q_emb`, `q_proj` | **Trainable** (random init, az answer loss) |

**Question conditioning dar captioner:**

```
qctx = mean_pool(q_emb(question_ids)) → q_proj   # (512)
attention_query = h_{t-1} + qctx                  # har decode step
```

### v_cap — do halat (`caption_repr`)

| Mode | Train | Eval | Gradient be q_emb? |
|------|-------|------|---------------------|
| `hidden` | Mean LSTM hidden (EOS-masked) | mean word_emb(tokens) | Yes (train) |
| `text` | Greedy tokens → `cap_txt_gru` | same | No (caption decode no_grad) |

<!--
  Baraye question-aware caption train, caption_repr: hidden behtar ast.
  hidden mode: grad az answer_loss → v_cap → h → qctx → q_emb.
-->

### Fusion (`fuse_mode`)

| Mode | Formula | `v` dimension |
|------|---------|---------------|
| `mul` | `v = v_cap ⊙ v_att` | 512 |
| `add` | `v = v_cap + v_att` | 512 |
| `concat` | `v = [v_cap ; v_att]` | 1024 |

### Answer decoder (Dual LSTM)

**Paper Eq. 10 — Attention LSTM:**
```
h1_t = LSTM_att( a_emb(a_{t-1}), g, h2_{t-1} )
```

**Paper Eq. 13 — Answer LSTM:**
```
h2_t = LSTM_ans( h1_t, h2_{t-1}, v, q_vec )
logit_t = Linear(h2_t)
```

<!--
  q_vec mostaghim vared lstm_ans mishe (na faghat az tarigh v_att).
  In baraye overfit roye sample haye kam kheili mohem bood.
-->

### Abaad — VQA

| Key | Default | Tozih |
|-----|---------|-------|
| `hidden_dim` | 512 | LSTM, projections |
| `word_dim` | 512 | q/a embeddings |
| `question_dim` | 1280 | GRU hidden |
| `max_regions` | 32 | ROI count |
| `max_question_len` | 14 | + BOS/EOS |
| `max_answer_len` | 6 | + BOS/EOS |

### Train (Stage 2)

```
Input:  (image, question_ids, answer_ids GT)
Loss:   CrossEntropy roye answer_ids[:, 1:]
Metric: VQA v2 soft accuracy
Output: outputs/<run>/best.pt  (VQAModel + q_vocab + a_vocab + captioner.q_emb)
```

---

## Do Vocabulary joda (mohem!)

<!--
  Ehtemal shtebah: caption vocab va question vocab ro qati kardim.
  word_emb captioner ≠ q_emb VQA ≠ q_emb captioner.
-->

| Embedding | Vocab | Size (smoke) | Koja estefade mishe |
|-----------|-------|--------------|---------------------|
| `captioner.word_emb` | MSCOCO caption | ~9906 | Caption decode |
| `captioner.q_emb` | VQA question | ~7650 | Caption question bias |
| `VQAModel.q_emb` | VQA question | ~7650 | v_att + answer LSTM |
| `VQAModel.a_emb` | VQA answer | ~12964 | Answer decode |

---

## File map

```
src/
├── architecture/
│   ├── ARCHITECTURE.en.md       English
│   └── ARCHITECTURE.fa.md       Finglish (in file)
├── SimpleImageCaptioner/
│   ├── train.py                 Stage 1 training
│   ├── eval.py                  Caption inference / demo
│   ├── models/
│   │   ├── captioner_v1.py      SimpleImageCaptioner (asli)
│   │   ├── relation_gnn.py      GNN baraye captioner
│   │   └── base_captioner.py    Interface
│   └── configs/smoke.yaml
│
└── SimpleVQA/
    ├── train.py                 Stage 2 training + VQAModel
    ├── eval.py                  VQA inference / accuracy
    ├── diagnose_caption_q.py    Test: aya soal caption ro taghir mide?
    └── configs/smoke.yaml
```

---

## Workflow — az train ta eval

```bash
# 1) Stage 1
cd SimpleImageCaptioner
python train.py --config configs/smoke.yaml

# 2) Stage 2 (captioner_ckpt ro to yaml set kon)
cd ../SimpleVQA
python train.py --config configs/smoke.yaml

# 3) Eval VQA
python eval.py --config configs/smoke.yaml --ckpt outputs/smoke/best.pt --split val --samples 20

# 4) Eval caption ba soal (question-guided)
cd ../SimpleImageCaptioner
python eval.py --config configs/smoke.yaml --ckpt outputs/smoke/best.pt \
  --vqa-ckpt ../SimpleVQA/outputs/smoke/best.pt \
  --image-id 25 --split train --question "How many animals are in this photo?"
```

---

## Cache (sari tar shodan train)

<!--
  Region va global feature ro 1 bar save mikonim ta har epoch
  Faster R-CNN / ResNet dobare run nashe.
-->

| Cache | Project | Path pattern | Content |
|-------|---------|--------------|---------|
| Region (captioner) | SimpleImageCaptioner | `{image_id}.pt` | raw ROI 1024-D |
| Region (VQA) | SimpleVQA | `{image_id}_k32_raw1024.pt` | raw ROI 1024-D |
| Global (VQA) | SimpleVQA | `{image_id}.pt` | ResNet 2048-D |

---

## References

- Sharma & Jalal (2021) — *Image captioning improved visual question answering*
- Xu et al. (2015) — Show, Attend and Tell (attention + LSTM init)
- VQA v2 dataset — Open-ended questions + 10 annotator answers
