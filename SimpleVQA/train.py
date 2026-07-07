"""Simple VQA — train va eval dar yek file (Paper: Image captioning improved VQA).

Paper / thesis (do marhale)
----------------------------
1. Aval captioner ro roye MSCOCO train kon (``SimpleImageCaptioner/train.py``).
2. Bad captioner ro freeze kon va dakhele ``VQAModel`` estefade kon.

Model flow (kholase)
--------------------
``image`` → ResNet global + Faster R-CNN regions → RelationGNN → ``v_att``
``question`` → GRU → ``q``
captioner (freeze) + ``q_ids`` → ``v_cap``
``v = v_cap * v_att`` (ya ``+`` ba ``fuse_mode``)
dual LSTM → javab

Run az ``SimpleVQA/``::

    python train.py --config configs/default.yaml
    python train.py --config configs/smoke.yaml
    python train.py --config configs/default.yaml --eval --ckpt outputs/default/best.pt
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import yaml
from PIL import Image
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adamax
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.models import ResNet101_Weights, resnet101
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from tqdm import tqdm

# Finglish: torchrun/Windows type hint compatibility
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
TOKEN_RE = re.compile(r"[a-z0-9']+")

"""
Finglish note (Kaggle 2xT4 + speed)
----------------------------------
- cache_regions: raw FasterRCNN region feature ha (max_regions, 1024) disk save mikonim.
  MOHEM: faqat khoroji box_head encoder cache mishe (1024d) — local_proj (1024→hidden_dim)
  CACHE NEMISHE va hamishe ejra mishe (trainable). Epoch haye badi kheili faster mishan.
  FasterRCNN(img) → 1024d [cache inja] → local_proj(1024→hidden_dim) [no cache, trains].
  train/val dir joda: train_region_cache_dir vs val_region_cache_dir.
- cache_global: raw ResNet-101 feature (pool+flatten → 2048d) disk save mikonim.
  MOHEM: faqat khoroji encoder cache mishe — g_proj (linear 2048→hidden_dim) CACHE NEMISHE
  va hamishe dar forward pass ejra mishe (trainable projection).
  ResNet101(img) → 2048d [cache inja] → g_proj(2048→hidden_dim) [no cache, trains].
  train/val dir joda: train_global_cache_dir vs val_global_cache_dir.
- use_amp: mixed precision baraye speed/memory.
- ddp: age ddp=true bashe, ba torchrun do ta GPU ro hamzaman estefade mikonim.
"""


def ddp_env() -> Tuple[bool, int, int, int]:
    """Finglish: DDP env ro az torchrun migirime (WORLD_SIZE/RANK/LOCAL_RANK)."""
    try:
        ws = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    except Exception:
        ws, rank, local_rank = 1, 0, 0
    return ws > 1, ws, rank, local_rank


def ddp_setup(cfg: Dict[str, Any]) -> Tuple[bool, int, int, int]:
    """Finglish: age ddp=true bashe process group ro init mikonim."""
    want = bool(cfg.get("ddp", False))
    enabled, world, rank, local_rank = ddp_env()
    if not want or not enabled:
        return False, 1, 0, 0
    backend = str(cfg.get("ddp_backend", "nccl"))
    dist.init_process_group(backend=backend, init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True, world, rank, local_rank


def unwrap_model(model: nn.Module) -> nn.Module:
    """Finglish: DDP wrapper ro bardarim — load/save state_dict roye model asli."""
    return model.module if hasattr(model, "module") else model


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config(path: str) -> Dict[str, Any]:
    """YAML config ro load kon (hyperparameter-ha va path dataset)."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path_fields(cfg: Dict[str, Any], keys: Tuple[str, ...]) -> None:
    """Path haye relative ro be absolute tabdil kon (in-place)."""
    for key in keys:
        value = cfg.get(key)
        if isinstance(value, str) and value:
            cfg[key] = str(Path(value).expanduser().resolve())


def set_seed(seed: int) -> None:
    """RNG seed baraye reproducibility (Python + Torch)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cap_list(items: List[int], cap: Optional[int]) -> List[int]:
    """Age cap > 0 bashe, faghat avvalin N ta item ro negah dar (smoke test)."""
    if cap is None or cap <= 0:
        return items
    return items[: int(cap)]


# ---------------------------------------------------------------------------
# Feature cache dirs (train / val joda — FasterRCNN + ResNet-101)
# ---------------------------------------------------------------------------
def region_cache_dir_for_split(cfg: Dict[str, Any], split: str) -> Optional[str]:
    """Finglish — path cache region FasterRCNN baraye train ya val.

    age ``cache_regions: false`` → None.
    train → ``train_region_cache_dir`` ; val → ``val_region_cache_dir``.
    """
    if not bool(cfg.get("cache_regions", False)):
        return None
    key = "train_region_cache_dir" if split == "train" else "val_region_cache_dir"
    return cfg.get(key)


def global_cache_dir_for_split(cfg: Dict[str, Any], split: str) -> Optional[str]:
    """Finglish — path cache global ResNet-101 baraye train ya val.

    age ``cache_global: false`` → None.
    train → ``train_global_cache_dir`` ; val → ``val_global_cache_dir``.
    """
    if not bool(cfg.get("cache_global", False)):
        return None
    key = "train_global_cache_dir" if split == "train" else "val_global_cache_dir"
    return cfg.get(key)


# ---------------------------------------------------------------------------
# Tokenizer & Vocab
# ---------------------------------------------------------------------------
def tok(text: str) -> List[str]:
    """Matn ro lowercase token kon (hamoon convention SimpleImageCaptioner)."""
    return TOKEN_RE.findall(text.lower())


class Vocab:
    """Vocab baraye soal/javab: PAD=0, BOS=1, EOS=2, UNK=3."""

    PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"

    def __init__(self, words: List[str], min_freq: int = 4) -> None:
        """Az list kalamat, vocab besaz; kalamat kam-frequency filter mishan."""
        counts = Counter(words)
        self.itos = [self.PAD, self.BOS, self.EOS, self.UNK] + sorted(
            w for w, n in counts.items() if n >= min_freq
        )
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, words: List[str]) -> List[int]:
        """Token list → index list (unknown → UNK)."""
        unk = self.stoi[self.UNK]
        return [self.stoi.get(w, unk) for w in words]

    @property
    def pad_id(self) -> int:
        return self.stoi[self.PAD]


def mode_answer(answers: List[str]) -> str:
    """Az 10 javab annotator, mode (por-tekrar-tarin) ro bargardoon."""
    return Counter(a.strip().lower() for a in answers).most_common(1)[0][0]


def all_qids(questions_json: str) -> List[int]:
    """Hame question_id haye yek split VQA v2 ro begir."""
    with Path(questions_json).open("r", encoding="utf-8") as f:
        qs = json.load(f)["questions"]
    return [int(x["question_id"]) for x in qs]


def build_vocabs(
    questions_json: str, annotations_json: str, min_freq: int
) -> Tuple[Vocab, Vocab]:
    """Vocab soal/javab ro faghat az train split besaz (leakage nabashe)."""
    with Path(questions_json).open("r", encoding="utf-8") as f:
        qs = json.load(f)["questions"]
    with Path(annotations_json).open("r", encoding="utf-8") as f:
        anns = json.load(f)["annotations"]
    qmap = {int(x["question_id"]): x for x in qs}
    amap = {int(x["question_id"]): x for x in anns}
    qids = sorted(set(qmap.keys()) & set(amap.keys()))
    q_words: List[str] = []
    a_words: List[str] = []
    for qid in qids:
        q_words.extend(tok(qmap[qid]["question"]))
        ans = [z["answer"] for z in amap[qid]["answers"]]
        a_words.extend(tok(mode_answer(ans)))
    return Vocab(q_words, min_freq=min_freq), Vocab(a_words, min_freq=1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class VQADataset(Dataset):
    """Yek sample = (image 448×448, soal token, javab token, 10 javab raw)."""

    def __init__(
        self,
        questions_json: str,
        annotations_json: str,
        images_dir: str,
        image_filename_template: str,
        q_vocab: Vocab,
        a_vocab: Vocab,
        max_q: int,
        max_a: int,
        qids: Optional[List[int]] = None,
        image_size: int = 448,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.image_filename_template = image_filename_template
        self.q_vocab = q_vocab
        self.a_vocab = a_vocab
        self.max_q = max_q
        self.max_a = max_a

        with Path(questions_json).open("r", encoding="utf-8") as f:
            qs = json.load(f)["questions"]
        with Path(annotations_json).open("r", encoding="utf-8") as f:
            anns = json.load(f)["annotations"]
        qmap = {int(x["question_id"]): x for x in qs}
        amap = {int(x["question_id"]): x for x in anns}
        use_qids = sorted(qids) if qids is not None else sorted(set(qmap.keys()) & set(amap.keys()))

        self.samples: List[Dict[str, Any]] = []
        for qid in use_qids:
            if qid not in qmap or qid not in amap:
                continue
            q = qmap[qid]
            a = amap[qid]
            answers = [x["answer"] for x in a["answers"]]
            self.samples.append(
                {
                    "qid": qid,
                    "image_id": int(q["image_id"]),
                    "question": q["question"],
                    "answers": answers,
                    "answer": mode_answer(answers),
                }
            )

        self.transform = transforms.Compose(
            [
                transforms.Resize((int(image_size), int(image_size))),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        path = self.images_dir / self.image_filename_template.format(image_id=s["image_id"])
        image = self.transform(Image.open(path).convert("RGB"))
        q_ids = [1] + self.q_vocab.encode(tok(s["question"])[: self.max_q - 2]) + [2]
        a_ids = [1] + self.a_vocab.encode(tok(s["answer"])[: self.max_a - 2]) + [2]
        return {
            "image": image,
            "image_id": int(s["image_id"]),
            "q": torch.tensor(q_ids, dtype=torch.long),
            "a": torch.tensor(a_ids, dtype=torch.long),
            "answers": s["answers"],
        }


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Batch ro stack kon; soal/javab ro ba PAD=0 padding bede."""
    images = torch.stack([x["image"] for x in batch])
    image_ids = torch.tensor([int(x["image_id"]) for x in batch], dtype=torch.long)
    max_q = max(len(x["q"]) for x in batch)
    max_a = max(len(x["a"]) for x in batch)
    q = torch.zeros((len(batch), max_q), dtype=torch.long)
    a = torch.zeros((len(batch), max_a), dtype=torch.long)
    for i, x in enumerate(batch):
        q[i, : len(x["q"])] = x["q"]
        a[i, : len(x["a"])] = x["a"]
    return {
        "images": images,
        "image_ids": image_ids,
        "q": q,
        "a": a,
        "answers": [x["answers"] for x in batch],
    }


# ---------------------------------------------------------------------------
# Captioner load — do vocabulary + fine-tune q_emb (marhale 2 VQA)
# ---------------------------------------------------------------------------
# Captioner roye MSCOCO caption train shode (bedoon soal).
# VQA soal dare → ``q_emb`` joda az ``word_emb``; faghat q_emb/q_proj trainable.
# ---------------------------------------------------------------------------

def _load_matching_state_dict(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    """Checkpoint ro load kon — faghat layer hayi ke shape-shoon match mikone.

    Vaghti caption vocab != question vocab, ``word_emb`` dige resize nemishe;
    in func baghiye weight ha (LSTM, attention, region encoder, ...) ro load
    mikone va key haye jadid mesl ``q_emb`` ke toye checkpoint nist ro skip mikone.
    """
    model_state = model.state_dict()
    filtered = {
        key: tensor
        for key, tensor in state.items()
        if key in model_state and model_state[key].shape == tensor.shape
    }
    skipped = [key for key in state if key not in filtered]
    if skipped:
        print(
            "Captioner checkpoint: skipped "
            f"{len(skipped)} key(s) with shape mismatch or unknown name "
            f"({', '.join(skipped)})"
        )
    model.load_state_dict(filtered, strict=False)


def _unfreeze_captioner_question_layers(captioner: nn.Module) -> int:
    """Hame captioner ro freeze kon, faghat ``q_emb`` va ``q_proj`` trainable bezan.

    Chera?
        - marhale 1 (caption train): soal feed nemishod → ``q_emb`` train nashode / random
        - marhale 2 (VQA): soal bayad caption ro **question-guided** kone
        - caption GT nadarim → faghat in 2 layer ro az **answer loss** (indirect) update mikonim
        - LSTM, attention, ``word_emb``, ``classifier`` freeze mimonan (knowledge caption train)

    Returns:
        Tedad parameter haye trainable (baraye log).
    """
    for param in captioner.parameters():
        param.requires_grad = False
    trainable = 0
    q_emb = getattr(captioner, "q_emb", None)
    if q_emb is not None:
        for param in q_emb.parameters():
            param.requires_grad = True
            trainable += param.numel()
    q_proj = getattr(captioner, "q_proj", None)
    if q_proj is not None and not isinstance(q_proj, nn.Identity):
        for param in q_proj.parameters():
            param.requires_grad = True
            trainable += param.numel()
    return trainable


def _caption_vocab_size_from_checkpoint(ckpt_path: Path) -> int:
    """Tedad kalamat vocabulary caption ro az checkpoint captioner begir.

    Az ``state['vocab']`` ya shape ``word_emb.weight`` estefade mikone.
    """
    state = torch.load(ckpt_path, map_location="cpu")
    vocab = state.get("vocab")
    if vocab is not None:
        return len(vocab)
    weight = state.get("model", state).get("word_emb.weight")
    if weight is not None:
        return int(weight.shape[0])
    raise ValueError(f"Cannot infer caption vocabulary size from {ckpt_path}")


def load_captioner(
    cfg: Dict[str, Any], q_vocab_size: int, pad_id: int, device: torch.device
) -> nn.Module:
    """Captioner pretrained ro load kon va freeze kon (marhale 2 VQA).

    Design:
        - ``word_emb`` / ``classifier`` → caption vocab az checkpoint (mesl smoke: 249)
        - ``q_emb`` → question vocab VQA (mesl smoke: 7650) — random init
        - ``q_emb`` + ``q_proj`` trainable; baghiye captioner freeze
        - train: grad answer loss → v_cap (LSTM hidden pool) → ``q_emb``
        - eval: v_cap az ``word_emb`` caption tolid shode (paper §3.4)

    Args:
        cfg: path haye ``captioner_project_root``, ``captioner_ckpt``, hyperparams.
        q_vocab_size: ``len(q_vocab.itos)`` — size vocabulary soal.
        pad_id: PAD index (0).
        device: cuda/cpu.
    """

    captioner_root = Path(cfg["captioner_project_root"]).resolve()

    # add captioner project to python path
    if str(captioner_root) not in sys.path:
        sys.path.insert(0, str(captioner_root))

    module_name = "models.captioner_v1"

    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        raise ImportError(f"Cannot import captioner module {module_name}: {e}")

    cls = getattr(mod, cfg.get("captioner_class", "SimpleImageCaptioner"))

    ckpt_path = Path(cfg["captioner_ckpt"])
    if ckpt_path.exists():
        caption_vocab_size = _caption_vocab_size_from_checkpoint(ckpt_path)
    else:
        caption_vocab_size = q_vocab_size
        print(
            f"Captioner checkpoint not found at {ckpt_path}; "
            f"using question vocab size {q_vocab_size} for caption layers."
        )

    model = cls(
        vocab_size=caption_vocab_size,
        pad_id=pad_id,
        word_dim=int(cfg["word_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        max_regions=int(cfg["max_regions"]),
        question_dim=int(cfg["question_dim"]),
        question_vocab_size=q_vocab_size,
        question_pad_id=pad_id,
    )

    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu")
        _load_matching_state_dict(model, state.get("model", state))
        print(
            f"Captioner loaded: caption_vocab={caption_vocab_size} "
            f"question_vocab={q_vocab_size}"
        )

    model.eval().to(device)

    trainable = _unfreeze_captioner_question_layers(model)
    if trainable:
        print(f"Captioner: {trainable} trainable params in q_emb + q_proj (rest frozen)")

    return model

# ---------------------------------------------------------------------------
# VQA Model (Paper §3.4 — caption-augmented answering)
# ---------------------------------------------------------------------------
class RelationGNN(nn.Module):
    """
    Message passing roye region-ha (relational reasoning).

    Yek Graph Neural Network sade baraye relational reasoning roye region-haye image.

    In module, har region ba hame region-haye dige interact mikone ta
    relation-haye object/object va context-haye vizhual behtar model beshan.
    voroudi x be soorat (batch, num_regions, dim) ast va khoroji hamoon
    region-feature-haye update shode mibashad.
    input: [N*32*2048]
    output: [N*32*2048] update shode(feature haye hamsaye ro ham be in region rabt mide)

    Flow:
        - for every pair of regions, a relation message sakhte mishavad.
        - edge messages aggregate mishavad.
        - node update final روی هر region اعمال می‌شود

    Inja idea in ast ke model faghat feature mokhtasar har region ra nabinad,
    balke dependencies beyn region-ha ra ham yad begirad; mesl object interaction,
    spatial relation, va contextual cue-ha.

    # Tareef
    node = region
    edge = relation between two regions
    pas edge(i,j) = relation(region_i , region_j)

    # Mesal
    x_i = (2048)
    x_j = (2048)
    concat mikonim: [x_i , x_j] = 4096
    edge_ij = Linear(4096 → 2048)

    mesal baraye dark behtar. tasvir shamel:
    person
    bicycle
    dog
    ball
    tree
    Relation hayi ke mitonan beyn har region(Node) dashte bashan:
    (person , bicycle) → riding
    (person , dog) → walking
    (dog , ball) → chasing
    """


    def __init__(self, dim: int = 512, dropout: float = 0.3) -> None:
        super().__init__()
        """
        Edge va node MLP-ha baraye message passing rooye region embeddings.
        degat kon inja graph fully connected hast. yani fagat toye hamsaye ha donbal relation nist. 

        - edge: relation between every pair of regions ra encode mikone (ertebaat beyn har region ro encode mikone)
        - node: message aggregated shode ra ba feature asli har node combine mikone

        dim: dimensionality-e representation-e har region.
        dropout: baade ReLU va qabl az linear dovvom — jelo overfit ro migirad.

        input: 512 + 512 (engar dota region ro concat mikone va be onvan vouroudi migire)
        output: 512 (va ye representation jadid az tarkib hardo voroudi misaze va be onvan khorouji mide)
        """
        # Finglish: Dropout baade ReLU mizarim ta GNN be rahat overfit nakone
        # (fully-connected graph O(K^2) edge dare — kheili high-capacity ast).
        self.edge = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, dim)
        )
        self.node = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        RelationGNN forward pass.

        Args:
            x (Tensor): Region features with shape (batch, num_regions, dim).
            Here each region is a node in the graph.

        Steps:
            For each region i and every other region j:
            1) xi = feature of region i, xj = feature of region j
            2) Concatenate them: [xi, xj] → relation input
            3) edge MLP computes relation embedding edge(i,j)
            4) For each region i, aggregate relations with all j (mean) -> engar beyn edge representation tamami hamsaye ha mean migirim.
            5) Concatenate original node feature x_i with the aggregated
            message and update it using the node MLP.
             (feature haye region i ro ba mean tamami edge hash concat mikonim va ye representation jadid az region feature i bedast miyarim)

        Returns:
            Tensor: Updated region features (batch, num_regions, dim).

        Example:
            If num_regions = 3 → regions {0,1,2}:
            relations computed: (0,1), (0,2), (1,0), (1,2), (2,0), (2,1).
        """
        b, k, d = x.shape
        xi = x.unsqueeze(2).expand(b, k, k, d)
        xj = x.unsqueeze(1).expand(b, k, k, d)
        edge_msg = self.edge(torch.cat([xi, xj], dim=-1)).mean(dim=2)
        return self.node(torch.cat([x, edge_msg], dim=-1))


class VQAModel(nn.Module):
    """
    VQA ba caption freeze + dual LSTM decoder.

    Main VQA model baraye answer generation ba tarkibe:
    - global visual feature az ResNet
    - local region feature az Faster R-CNN
    - relational reasoning az RelationGNN
    - question encoding ba GRU
    - caption-based representation az captioner freeze shode
    - dual LSTM decoder baraye answer generation

    Tebg paper (section 3.4), caption representation baed az generate shodan caption
    be onvan yek semantic image embedding estefade mishavad ta
    attended visual feature ra takmil konad.
    In model, v_att az region attention miayad, (v_att= hasel attention question roye region haye tasvir)
    v_cap az captioner frozen bedast avorde mishe,
    va ba fusion mode (mul ya add)
    edgam mishavand.
    fused_visual_representation= fuse(v_cap,v_att)


    Architecture summary:
        image -> global CNN feature + region proposals -> relation reasoning -> v_att
        question -> embedding + GRU -> q_vec
        image + question -> frozen captioner -> v_cap
        fused visual representation -> dual LSTM -> answer tokens

    Notes:
        - captioner completely frozen ast
        - detector ham freeze ast
        - fusion mode mishe 'mul' ya 'add'


    Pipeline:
    step 1:
        line 1:
            image
            ↓
            FasterRCNN
            ↓
            region features: [32* 2048]
            ↓
            RelationGNN: [32* 2048] khourji abaadesh fargi nemikone
            ↓
            attention with question
            ↓
            v_att: [1*2048]


        line 2:
            image
            ↓
            frozen caption model
            ↓
            generated caption
            ↓
            caption embedding
            ↓
            v_cap
        
    step2:
        v_att + v_cap
            ↓
        fused visual feature
            ↓
        dual LSTM
            ↓
        answer

    """
    def __init__(
        self,
        q_vocab_size: int,
        a_vocab_size: int,
        pad_id: int,
        captioner: Optional[nn.Module],
        word_dim: int = 512,
        hidden_dim: int = 512,
        question_dim: int = 1280,
        max_regions: int = 32,
        fuse_mode: str = "mul",
        dropout: float = 0.3,
        use_captioner: bool = True,
        caption_repr: str = "hidden",
    ) -> None:
        super().__init__()
        self.use_captioner = use_captioner
        if use_captioner and captioner is None:
            raise ValueError("use_captioner=True vali captioner=None — load_captioner() ro seda bezan.")
        if not use_captioner and captioner is not None:
            raise ValueError("use_captioner=False vali captioner dade shode — faghat yeki ro entekhab kon.")
        self.captioner = captioner
        self.max_regions = max_regions
        self.hidden_dim = hidden_dim
        self.fuse_mode = fuse_mode
        # Finglish — fuse_mode validation (paper Eq. 12 + ablation §5):
        #   "mul"    → element-wise multiply (matn-e maghale Eq. 12)
        #   "add"    → element-wise sum (ablation §5)
        #   "concat" → concatenation (ablation §5, behtarin natije-ye maghale)
        # Har chize dige → error, ta config ghalat silent nashe.
        if fuse_mode not in ("mul", "add", "concat"):
            raise ValueError(
                f"fuse_mode bayad 'mul' | 'add' | 'concat' bashe, na '{fuse_mode}'"
            )
        # ------------------------------------------------------------------
        # caption_repr — how the caption becomes v_cap
        # ------------------------------------------------------------------
        # EN: Two ways to build v_cap from the frozen captioner:
        #       "hidden" (default) → mean of the caption LSTM hidden states.
        #                 Faithful to the current code; carries little NEW info
        #                 because it re-pools the SAME frozen region features and
        #                 the generated caption TEXT is discarded.
        #       "text"   → generate the caption tokens (frozen, greedy) and read
        #                 them with a small TRAINABLE GRU over the captioner's word
        #                 embeddings. This injects the caption's semantic content as
        #                 real text (paper Sharma & Jalal §3.4) and gives the model a
        #                 trainable channel to actually use captions.
        # FA: Do ravesh baraye sakht-e v_cap az captioner-e frozen:
        #       "hidden" (pishfarz) → miangin hidden-state-haye LSTM-e caption.
        #                 Etela'at-e jadid kam dare chon hamun region-feature-haye
        #                 frozen ro dobare pool mikone va matn-e caption dor rikhte mishe.
        #       "text"   → token-haye caption ro (frozen, greedy) tolid mikone va ba
        #                 yek GRU-ye TRAINABLE roye word-embedding-e captioner mikhune.
        #                 Ma'na-ye caption ro be onvan matn-e vaghei tazrigh mikone
        #                 (maghale §3.4) va yek masir-e trainable baraye estefade az caption mide.
        if caption_repr not in ("hidden", "text"):
            raise ValueError(
                f"caption_repr bayad 'hidden' ya 'text' bashe, na '{caption_repr}'"
            )
        self.caption_repr = caption_repr
        """
        Constructor for multimodal VQA pipeline.

        Parameters:
            q_vocab_size: size of question vocabulary
            a_vocab_size: size of answer vocabulary
            pad_id: padding token id for embeddings
            captioner: pretrained image captioning model, frozen during VQA training
            word_dim: embedding size for question/answer tokens
            hidden_dim: shared hidden dimension for visual and answer modules
            question_dim: GRU output dimension for question encoding
            max_regions: maximum number of region proposals kept from detector
            fuse_mode: how v_cap and v_att are fused ('mul' or 'add')
            dropout: dropout probability applied after all trainable projections and LSTM outputs
        """
        # Finglish: yek Dropout module baraye kol model — probability az config miad.
        # dar eval mode (.eval()) PyTorch khod Dropout ro غیرفعال mikone.
        self.drop = nn.Dropout(dropout)

        backbone = resnet101(weights=ResNet101_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.g_proj = nn.Linear(2048, hidden_dim)
        
        # mige weight haye Resnet garar nist update beshe.
        for p in self.resnet.parameters():
            p.requires_grad = False

        detector = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        self.detector = detector

        # mige weight hash garar nist update beshe.
        for p in self.detector.parameters():
            p.requires_grad = False

        self.local_proj = nn.Linear(1024, hidden_dim)

        self.q_emb = nn.Embedding(q_vocab_size, word_dim, padding_idx=pad_id)
        self.q_gru = nn.GRU(word_dim, question_dim, batch_first=True)
        self.q_proj = nn.Linear(question_dim, hidden_dim)

        self.gnn = RelationGNN(hidden_dim, dropout=dropout)
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attn_score = nn.Linear(hidden_dim, 1)

        self.a_emb = nn.Embedding(a_vocab_size, word_dim, padding_idx=pad_id)

        # LSTM_att (paper Eq. 10): h1_t = LSTM_att(a_{t-1}, h1_{t-1}, [h2_{t-1}; vG])
        # Inputs concat: answer_embed(t-1) + global_visual_feature(vG=g) + h2(t-1).
        self.lstm_att = nn.LSTMCell(word_dim + hidden_dim + hidden_dim, hidden_dim)

        # Finglish — dimension-e feature-e fuse shode (v_ft) bastegi be use_captioner + fuse_mode dare:
        #   use_captioner=False → v = v_att → hidden_dim
        #   use_captioner=True, mul/add → hidden_dim
        #   use_captioner=True, concat    → hidden_dim * 2
        if use_captioner and fuse_mode == "concat":
            fused_dim = hidden_dim * 2
        else:
            fused_dim = hidden_dim

        # LSTM_ans (paper Eq. 13): h2_t = LSTM_ans(h1_t, h2_{t-1}, v_ft, q)
        # CONFLICT FIX: question vector `q` mostaghim be answer LSTM dade mishe.
        #   Ghablan `q` faghat gheyr-e mostaghim (az tarigh-e v_att) mi-rasid → answer decoder
        #   nemitoonest beyn chand soal-e yek tasvir tafrigh bede va roye 100 sample
        #   train_acc dar ~0.40 gir mikard. Tebghe Eq. 13, `q` bayad voroudi-ye mostaghim
        #   -e LSTM_ans bashe. Inputs concat: h1_t + h2_{t-1} + v_ft + q_vec.
        self.lstm_ans = nn.LSTMCell(
            hidden_dim + hidden_dim + fused_dim + hidden_dim, hidden_dim
        )

        self.out = nn.Linear(hidden_dim, a_vocab_size)

        # ------------------------------------------------------------------
        # Trainable caption-text encoder (only for caption_repr == "text")
        # ------------------------------------------------------------------
        # EN: Reads the generated caption tokens (embedded with the captioner's
        #     FROZEN word_emb) and encodes them into a hidden_dim vector v_cap.
        #     The captioner (generator) stays frozen; only THIS GRU trains, giving
        #     the model a learnable way to consume caption text. Output dim is
        #     hidden_dim, so fuse_mode ("mul"/"add"/"concat") logic is unchanged.
        # FA: Token-haye caption-e tolid-shode ro (ba word_emb-e FROZEN-e captioner
        #     embed shode) mikhune va be yek vector-e hidden_dim (v_cap) tabdil mikone.
        #     Khod-e captioner frozen mimune; faghat HAMIN GRU train mishe ta model
        #     betune az matn-e caption estefade kone. Khoruji hidden_dim ast, pas
        #     mantegh-e fuse_mode taghir nemikone.
        self.cap_txt_gru: Optional[nn.GRU] = None
        if use_captioner and caption_repr == "text":
            self.cap_txt_gru = nn.GRU(word_dim, hidden_dim, batch_first=True)

    def _regions_cache_path(self, cache_dir: str, image_id: int) -> Path:
        """
        Finglish: path fayle cache baraye yek image_id barmigardonad.
        Filename: {image_id}_k{max_regions}_raw1024.pt
        Chon raw FasterRCNN output hamishe 1024d ast (fixed by box_head),
        niazi be encode kardan dimension nist — faqat max_regions encode mishe.
        (qabl az in _d{hidden_dim} encode mishod ke ghalat bood — projected dim cache mishod)
        """
        return Path(cache_dir) / f"{int(image_id)}_k{int(self.max_regions)}_raw1024.pt"

    def _load_regions_cached(
        self, cache_dir: str, image_ids: torch.Tensor, device: torch.device
    ) -> Optional[torch.Tensor]:
        """
        Finglish: raw FasterRCNN region feature ha ro az cache disk load mikone.
        Age hatta yek image cache nadasht ya shape ghalat bood, None bar migardone
        ta kol batch dobare hesab beshe.

        Shape validation: (max_regions, 1024) — raw box_head output qabl az local_proj.
        (1024 fixed ast chon box_head architecture fix ast)

        Output:
            tensor (B, max_regions, 1024) age hame cache vojood dasht, vagharno None
        """
        try:
            paths = [self._regions_cache_path(cache_dir, int(i.item())) for i in image_ids]
            if not all(p.exists() for p in paths):
                return None
            tensors: List[torch.Tensor] = []
            for p in paths:
                t = torch.load(p, map_location="cpu")
                if not isinstance(t, torch.Tensor) or t.ndim != 2:
                    return None
                # BUG FIX (dtype): cache momkene fp16 zakhire shode bashe (AMP run).
                #   Bedoon-e .float(), vaghti AMP khamush ast, local_proj (fp32) ba
                #   voroudi-ye Half error mide ("mat1 and mat2 must have same dtype").
                #   Hamun kari ke RegionEncoder-e captioner ham mikone.
                tensors.append(t.float())
            out = torch.stack(tensors, dim=0)
            # shape: (B, max_regions, 1024) — raw encoder output, NOT projected
            if out.shape[1] != self.max_regions or out.shape[2] != 1024:
                return None
            return out.to(device, non_blocking=(device.type == "cuda"))
        except Exception:
            return None

    def _save_regions_cached(self, cache_dir: str, image_ids: torch.Tensor, regions: torch.Tensor) -> None:
        """
        Finglish: raw FasterRCNN region feature haro baraye har image disk save mikone.
        Har file yek tensor (max_regions, 1024) ast — raw box_head output QABL az local_proj.
        local_proj (trainable linear) cache NEMISHE va hamishe ejra mishe.
        Age hata yek error bashe, silent pass mikone ta train interrupt nashe.
        """
        try:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            for i, img_id in enumerate(image_ids):
                p = self._regions_cache_path(cache_dir, int(img_id.item()))
                torch.save(regions[i].detach().to("cpu"), p)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # ResNet-101 global feature cache
    # ------------------------------------------------------------------

    def _global_cache_path(self, cache_dir: str, image_id: int) -> Path:
        """
        Finglish: path fayle cache baraye yek image_id barmigardonad.
        Filename: {image_id}_resnet101.pt
        Chon ResNet-101 hamishe 2048d output dare, niazi be encode kardan dimension nist.
        """
        return Path(cache_dir) / f"{int(image_id)}_resnet101.pt"

    def _load_global_cached(
        self, cache_dir: str, image_ids: torch.Tensor, device: torch.device
    ) -> Optional[torch.Tensor]:
        """
        Finglish: cache disk ro baray yek batch image load mikone.
        Age hatta yek image cache nadasht ya shape ghalat bood, None bar migardone
        ta kol batch dobare hesab beshe (hamoon ravesh _load_regions_cached).

        Output:
            tensor (B, 2048) age hame cache vojood dasht, vagharno None
        """
        try:
            paths = [self._global_cache_path(cache_dir, int(i.item())) for i in image_ids]
            if not all(p.exists() for p in paths):
                return None
            tensors: List[torch.Tensor] = []
            for p in paths:
                t = torch.load(p, map_location="cpu")
                if not isinstance(t, torch.Tensor) or t.ndim != 1 or t.shape[0] != 2048:
                    return None
                # BUG FIX (dtype): cache momkene fp16 bashe (AMP). Bedoon-e .float()
                #   ba AMP khamush, g_proj (fp32) error mide. Hamishe fp32 bar migardoonim.
                tensors.append(t.float())
            out = torch.stack(tensors, dim=0)  # (B, 2048)
            return out.to(device, non_blocking=(device.type == "cuda"))
        except Exception:
            return None

    def _save_global_cached(
        self, cache_dir: str, image_ids: torch.Tensor, feats: torch.Tensor
    ) -> None:
        """
        Finglish: ResNet-101 raw feature haro baraye har image disk save mikone.
        Har file yek tensor (2048,) ast — qabl az g_proj linear layer.
        Age hata yek error bashe, silent pass mikone ta train interrupt nashe.
        """
        try:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            for i, img_id in enumerate(image_ids):
                p = self._global_cache_path(cache_dir, int(img_id.item()))
                torch.save(feats[i].detach().to("cpu"), p)
        except Exception:
            pass

    @torch.no_grad()
    def _global_feat(
        self,
        images: torch.Tensor,
        image_ids: Optional[torch.Tensor] = None,
        cache_dir: Optional[str] = None,
        save_cache: bool = True,
    ) -> torch.Tensor:
        """
        Finglish: ResNet-101 global feature ro extract ya az cache load mikone.

        Mohem:
            - Faqat khoroji ResNet+pool ro cache mikonim (2048d raw) — QABL az g_proj.
            - g_proj (linear layer) cache nemishe va dar har forward pass ejra mishe.
            - Agar cache_dir set bashe va image_ids vojood dashte bashe:
                1. Aval cache disk ro check mikonim.
                2. Age peyda nashd, ResNet ro ejra mikonim va save mikonim (dar train).
            - ResNet freeze ast pas @torch.no_grad() safe ast.

        Output:
            tensor (B, 2048) — raw pooled ResNet feature, QABL az linear projection
        """
        device = images.device
        if cache_dir and image_ids is not None and image_ids.numel() == images.size(0):
            cached = self._load_global_cached(cache_dir, image_ids, device)
            if cached is not None:
                return cached

        # ResNet-101 forward: (B,3,H,W) → (B,2048,h,w) → pool → (B,2048)
        feat = self.pool(self.resnet(images)).flatten(1)

        if cache_dir and image_ids is not None and image_ids.numel() == images.size(0) and save_cache:
            self._save_global_cached(cache_dir, image_ids, feat)

        return feat

    def train(self, mode: bool = True) -> "VQAModel":
        """Train VQA layers vali ResNet-101 va Faster R-CNN hamishe eval negah dar.

        RPN dar halat train error ``targets should not be None`` mide chon label
        nadarim — detector freeze ast. ResNet-101 ham freeze ast (pretrained global
        feature). captioner eval mimune (BN freeze); faghat ``q_emb`` / ``q_proj``
        trainable hastan va gradient migirand.
        """
        super().train(mode)
        self.resnet.eval()
        self.detector.eval()
        if self.captioner is not None:
            self.captioner.eval()
        return self

    def _regions(
        self,
        images: torch.Tensor,
        image_ids: Optional[torch.Tensor] = None,
        cache_dir: Optional[str] = None,
        save_cache: bool = True,
    ) -> torch.Tensor:
        """
        Finglish: FasterRCNN region feature ha ro extract ya az cache load mikone,
        bad ba local_proj be hidden_dim project mikone.

        Mohem (cache boundary):
            - Faqat raw box_head output (max_regions, 1024) cache mishe — QABL az local_proj.
            - local_proj (Linear 1024→hidden_dim) CACHE NEMISHE va hamishe ejra mishe
              ta gradient ha be in layer beresand (trainable projection).
            - @torch.no_grad() faqat dakhel func roye FasterRCNN parts estefade mishe,
              na roye kol method — chon local_proj bayad gradient dashtee bashe.

        Flow:
            cache hit  → (B, max_regions, 1024) load → local_proj → (B, max_regions, hidden_dim)
            cache miss → FasterRCNN (no_grad) → raw 1024d → save cache → local_proj → return

        Output:
            tensor (B, max_regions, hidden_dim)
        """
        device = images.device

        # Aval cache raw 1024d ro check mikonim; age peyda shod local_proj ro rosh ejra mikonim.
        if cache_dir and image_ids is not None and image_ids.numel() == images.size(0):
            cached = self._load_regions_cached(cache_dir, image_ids, device)
            if cached is not None:
                # cached: (B, max_regions, 1024) — local_proj trainable, hamishe ejra mishe
                return self.local_proj(cached)

        # FasterRCNN freeze ast — no_grad faqat baraye in bakhsh
        with torch.no_grad():
            transformed, _ = self.detector.transform(list(images), None)
            feats = self.detector.backbone(transformed.tensors)
            props, _ = self.detector.rpn(transformed, feats, None)
            roi = self.detector.roi_heads.box_roi_pool(feats, props, transformed.image_sizes)
            roi = self.detector.roi_heads.box_head(roi)  # (total_regions, 1024) raw output
            counts = [len(p) for p in props]
            chunks = torch.split(roi, counts)
            padded = []
            for chunk in chunks:
                chunk = chunk[: self.max_regions]
                if chunk.size(0) < self.max_regions:
                    pad = torch.zeros(
                        (self.max_regions - chunk.size(0), chunk.size(1)), device=chunk.device
                    )
                    chunk = torch.cat([chunk, pad], dim=0)
                padded.append(chunk)
            raw = torch.stack(padded, dim=0)  # (B, max_regions, 1024) — raw encoder output

        # Cache raw 1024d QABL az local_proj
        if cache_dir and image_ids is not None and image_ids.numel() == images.size(0) and save_cache:
            self._save_regions_cached(cache_dir, image_ids, raw)

        # local_proj trainable ast — kharij az no_grad ejra mishe ta gradient beresad
        return self.local_proj(raw)

    def _attend(self, regions: torch.Tensor, q_vec: torch.Tensor) -> torch.Tensor:
        """
        Softmax attention roye region-ha ba vector soal.

        Har region ba q_vec compare mishavad, attention weight ha hesab mishavand,
        va yek attended visual vector khoroji mide ke namayande relevant parts image ast.

        input: [32* 2048]
        output: [1*2048]
        """
        b, k, d = regions.shape
        q_exp = q_vec.unsqueeze(1).expand(b, k, d)
        hidden = torch.tanh(self.attn(torch.cat([regions, q_exp], dim=-1)))
        weights = torch.softmax(self.attn_score(hidden).squeeze(-1), dim=-1)
        return torch.einsum("bk,bkd->bd", weights, regions)

    def forward(
        self,
        images: torch.Tensor,
        q_ids: torch.Tensor,
        a_ids: Optional[torch.Tensor] = None,
        max_answer_len: int = 6,
        image_ids: Optional[torch.Tensor] = None,
        region_cache_dir: Optional[str] = None,
        global_cache_dir: Optional[str] = None,
        save_cache: bool = True,
    ) -> torch.Tensor:
        """
        Train: ``a_ids`` bede (teacher forcing). Eval: ``a_ids=None`` (greedy).

        Forward pass for training ya inference.

        Train mode:
            - a_ids داده می‌شود
            - teacher forcing active ast
            - ``get_caption_embedding(..., differentiable=True)`` → grad be captioner.q_emb

        Eval mode:
            - a_ids=None
            - greedy decoding estefade mishavad
            - v_cap ham az hamun hidden-state mean (differentiable=True) hesab mishe
              (dakhel-e torch.no_grad), pas train va eval yek namayesh-e v_cap darand
              (paper §3.4). q_vec ham mostaghim vared-e LSTM_ans mishe (paper Eq. 13).

        Output:
            logits ba shape (batch, answer_len-1, a_vocab_size)
        """
        # ResNet-101 raw feature az cache ya live hesab mikonim (2048d).
        # g_proj linear layer CACHE NEMISHE va hamishe ejra mishe (trainable projection).
        raw_g = self._global_feat(
            images,
            image_ids=image_ids,
            cache_dir=global_cache_dir,
            save_cache=save_cache,
        )
        # [dropout 1/6] baade g_proj — global visual feature qabl az LSTM
        g = self.drop(self.g_proj(raw_g))

        # local feature haye img ro extract mikone.
        local = self._regions(
            images,
            image_ids=image_ids,
            cache_dir=region_cache_dir,
            save_cache=save_cache,
        )
        # [dropout 2/6] baade local_proj — region feature ha qabl az GNN
        local = self.drop(local)

        # question feature ba estefade az GRU extract karde.
        _, h = self.q_gru(self.q_emb(q_ids))
        # [dropout 3/6] baade q_proj — question vector qabl az attention
        q_vec = self.drop(self.q_proj(h[-1]))

        # local feature haro mide be GNN ta relation region haro toye feature hash emal kone.
        # GNN dakhel khod ham Dropout dare (baade ReLU dar edge va node MLP).
        rel = self.gnn(local)

        # attention mizanim beyn image_regions va question
        # [dropout 4/6] baade attention — attended visual vector qabl az fusion
        v_att = self.drop(self._attend(rel, q_vec))

        # Age estefade az captioner enabled bashe miyaym v_att va v_cap ro be raveshi ke taeen shode fuse mikonim.
        # age estefade az captioner enabled nist bashe miyaym fagat az v_att 
        # (question dependent image feature) estefade mikonim.
        if self.use_captioner:
            # --------------------------------------------------------------
            # v_cap — caption representation (two modes, see __init__)
            # --------------------------------------------------------------
            # EN: "text" mode generates caption tokens once (frozen, greedy, reusing
            #     the region cache) and encodes them with the trainable GRU → the
            #     caption's semantic content enters as real text. "hidden" mode keeps
            #     the original behaviour (mean of caption LSTM hidden states, now
            #     EOS-masked inside the captioner).
            # FA: Halat-e "text" token-haye caption ro yek-bar (frozen, greedy, ba
            #     estefade az region cache) tolid mikone va ba GRU-ye trainable encode
            #     mikone → ma'na-ye caption be onvan matn vared mishe. Halat-e "hidden"
            #     hamun raftar-e ghabli (miangin hidden-state, hala EOS-mask shode).
            if self.caption_repr == "text":
                with torch.no_grad():
                    cap_ids, _ = self.captioner._decode_caption(
                        images,
                        q_ids,
                        max_len=20,
                        collect_hidden=False,
                        image_ids=image_ids,
                        region_cache_dir=region_cache_dir,
                    )
                cap_emb = self.captioner.word_emb(cap_ids)
                _, h_cap = self.cap_txt_gru(cap_emb)
                v_cap = self.drop(h_cap[-1])
            else:
                # CONFLICT/BUG FIX (paper §3.4): v_cap = miangin hidden-state LSTM captioner.
                v_cap, _ = self.captioner.get_caption_embedding(
                    images,
                    q_ids,
                    differentiable=True,
                    image_ids=image_ids,
                    region_cache_dir=region_cache_dir,
                )
            if self.fuse_mode == "concat":
                v = torch.cat([v_cap, v_att], dim=-1)
            elif self.fuse_mode == "add":
                v = v_cap + v_att
            else:
                v = v_cap * v_att
        else:
            v = v_att

        batch = images.size(0)
        h1 = torch.zeros((batch, self.hidden_dim), device=images.device)
        c1 = torch.zeros_like(h1)
        h2 = torch.zeros_like(h1)
        c2 = torch.zeros_like(h1)

        if a_ids is None:
            steps = max_answer_len - 1
            prev = torch.full((batch,), 1, dtype=torch.long, device=images.device)
        else:
            steps = a_ids.size(1) - 1
            prev = a_ids[:, 0]

        logits: List[torch.Tensor] = []
        for t in range(steps):
            a_prev = self.a_emb(prev)

            # input haye attentsion_LSTM(answer_embed(t-1) + ans_LSTM_h(t-1) + img_global_feature)
            h1, c1 = self.lstm_att(torch.cat([a_prev, g, h2], dim=-1), (h1, c1))

            # [dropout 5/6] baade lstm_att — h1 qabl az voroodi be lstm_ans
            # paper Eq. 13: LSTM_ans(h1_t, h2_{t-1}, v_ft, q)
            #   voroudi = [h1_t (drop shode), h2_{t-1}, v_ft, q_vec]
            #   q_vec dar hame step-ha yeksan ast (question representation-e sabet).
            h2, c2 = self.lstm_ans(
                torch.cat([self.drop(h1), h2, v, q_vec], dim=-1), (h2, c2)
            )

            # [dropout 6/6] baade lstm_ans — h2 qabl az classifier
            logit = self.out(self.drop(h2))
            logits.append(logit)
            prev = logit.argmax(dim=-1) if a_ids is None else a_ids[:, t + 1]

        return torch.stack(logits, dim=1)


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------
def vqa_acc(pred: torch.Tensor, gts: List[List[str]], vocab: Vocab) -> float:
    """Soft VQA v2 accuracy: min(agreement/3, 1) roye 10 javab annotator."""
    score = 0.0
    for pred_row, answers in zip(pred.tolist(), gts):
        text = " ".join(
            vocab.itos[i] for i in pred_row if i < len(vocab.itos) and i > 2
        ).strip().lower()
        agree = sum(1 for a in answers if a.strip().lower() == text)
        score += min(agree / 3.0, 1.0)
    return score / max(1, len(gts))


# ---------------------------------------------------------------------------
# Build data + model
# ---------------------------------------------------------------------------
PATH_KEYS = (
    "train_questions_json",
    "train_annotations_json",
    "val_questions_json",
    "val_annotations_json",
    "train_images_dir",
    "val_images_dir",
    "captioner_project_root",
    "captioner_ckpt",
    "save_dir",
    "train_region_cache_dir",
    "val_region_cache_dir",
    "train_global_cache_dir",
    "val_global_cache_dir",
)


def build_loaders(
    cfg: Dict[str, Any],
) -> Tuple[Vocab, Vocab, DataLoader, DataLoader]:
    """Dataset train/val va DataLoader besaz."""
    q_vocab, a_vocab = build_vocabs(
        cfg["train_questions_json"],
        cfg["train_annotations_json"],
        int(cfg["vocab_min_freq"]),
    )
    tr_qids = cap_list(all_qids(cfg["train_questions_json"]), cfg.get("max_train_qids"))
    va_qids = cap_list(all_qids(cfg["val_questions_json"]), cfg.get("max_val_qids"))

    train_ds = VQADataset(
        cfg["train_questions_json"],
        cfg["train_annotations_json"],
        cfg["train_images_dir"],
        cfg["train_image_filename_template"],
        q_vocab,
        a_vocab,
        int(cfg["max_question_len"]),
        int(cfg["max_answer_len"]),
        qids=tr_qids,
        image_size=int(cfg.get("image_size", 448)),
    )
    val_ds = VQADataset(
        cfg["val_questions_json"],
        cfg["val_annotations_json"],
        cfg["val_images_dir"],
        cfg["val_image_filename_template"],
        q_vocab,
        a_vocab,
        int(cfg["max_question_len"]),
        int(cfg["max_answer_len"]),
        qids=va_qids,
        image_size=int(cfg.get("image_size", 448)),
    )

    device_is_cuda = cfg.get("device") == "cuda" and torch.cuda.is_available()
    loader_kw: Dict[str, Any] = {
        "batch_size": int(cfg["batch_size"]),
        "num_workers": int(cfg["num_workers"]),
        "collate_fn": collate_batch,
        "pin_memory": bool(cfg.get("pin_memory", False)) and device_is_cuda,
    }
    if int(cfg["num_workers"]) > 0:
        loader_kw["persistent_workers"] = bool(cfg.get("persistent_workers", False))
        loader_kw["prefetch_factor"] = int(cfg.get("prefetch_factor", 2))

    ddp_on, _, _, _ = ddp_env()
    want_ddp = bool(cfg.get("ddp", False)) and ddp_on
    train_sampler = DistributedSampler(train_ds, shuffle=True) if want_ddp else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if want_ddp else None
    train_loader = DataLoader(
        train_ds,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        **loader_kw,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        sampler=val_sampler,
        **loader_kw,
    )
    return q_vocab, a_vocab, train_loader, val_loader


def build_vqa_model(
    cfg: Dict[str, Any], q_vocab: Vocab, a_vocab: Vocab, device: torch.device
) -> Tuple[VQAModel, Optional[nn.Module]]:
    """VQAModel besaz; age ``use_captioner`` true bashe captioner ham load mishe."""
    use_captioner = bool(cfg.get("use_captioner", True))
    captioner: Optional[nn.Module] = None
    if use_captioner:
        captioner = load_captioner(cfg, len(q_vocab.itos), q_vocab.pad_id, device)
    else:
        print("use_captioner=false — v_att faghat (bedoon v_cap fusion) estefade mishe.")
    model = VQAModel(
        len(q_vocab.itos),
        len(a_vocab.itos),
        q_vocab.pad_id,
        captioner,
        int(cfg["word_dim"]),
        int(cfg["hidden_dim"]),
        int(cfg["question_dim"]),
        int(cfg["max_regions"]),
        str(cfg["fuse_mode"]),
        float(cfg.get("dropout", 0.3)),
        use_captioner=use_captioner,
        # EN: caption_repr chooses "hidden" (legacy) vs "text" (paper-faithful) v_cap.
        # FA: caption_repr beyn "hidden" (ghadimi) va "text" (motabegh-e maghale) entekhab mikone.
        caption_repr=str(cfg.get("caption_repr", "hidden")),
    ).to(device)
    return model, captioner


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------
def train_epoch(
    model: VQAModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    criterion: nn.Module,
    a_vocab: Vocab,
    cfg: Dict[str, Any],
    device: torch.device,
) -> Tuple[float, float]:
    """Yek epoch train — CE loss + batch ``vqa_acc``."""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    accum = int(cfg["grad_accum_steps"])
    use_amp = bool(cfg["use_amp"]) and device.type == "cuda"
    optimizer.zero_grad(set_to_none=True)

    for i, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        images = batch["images"].to(device, non_blocking=device.type == "cuda")
        image_ids = batch.get("image_ids")
        if image_ids is not None:
            image_ids = image_ids.to(device, non_blocking=device.type == "cuda")
        q = batch["q"].to(device, non_blocking=device.type == "cuda")
        a = batch["a"].to(device, non_blocking=device.type == "cuda")

        region_cache_dir = region_cache_dir_for_split(cfg, "train")
        global_cache_dir = global_cache_dir_for_split(cfg, "train")
        with autocast(enabled=use_amp):
            logits = model(
                images,
                q,
                a_ids=a,
                image_ids=image_ids,
                region_cache_dir=region_cache_dir,
                global_cache_dir=global_cache_dir,
                save_cache=True,
            )
            loss = criterion(logits.reshape(-1, logits.size(-1)), a[:, 1:].reshape(-1))

        scaler.scale(loss / accum).backward()
        if (i + 1) % accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.item())
        total_acc += vqa_acc(logits.argmax(dim=-1), batch["answers"], a_vocab)

    n = max(1, len(loader))
    return total_loss / n, total_acc / n


@torch.no_grad()
def eval_epoch(
    model: VQAModel,
    loader: DataLoader,
    criterion: nn.Module,
    a_vocab: Vocab,
    cfg: Dict[str, Any],
    device: torch.device,
    greedy: bool = False,
    split: str = "val",
) -> Tuple[float, float]:
    """Validation — teacher forcing ya greedy decode (``greedy=True``)."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    # Finglish: val split cache joda az train ; miss → save (mesl train loop).
    region_cache_dir = region_cache_dir_for_split(cfg, split)
    global_cache_dir = global_cache_dir_for_split(cfg, split)

    for batch in tqdm(loader, desc="val", leave=False):
        images = batch["images"].to(device, non_blocking=device.type == "cuda")
        image_ids = batch.get("image_ids")
        if image_ids is not None:
            image_ids = image_ids.to(device, non_blocking=device.type == "cuda")
        q = batch["q"].to(device, non_blocking=device.type == "cuda")
        a = batch["a"].to(device, non_blocking=device.type == "cuda")

        if greedy:
            logits = model(
                images,
                q,
                a_ids=None,
                max_answer_len=int(cfg["max_answer_len"]),
                image_ids=image_ids,
                region_cache_dir=region_cache_dir,
                global_cache_dir=global_cache_dir,
                save_cache=True,
            )
        else:
            logits = model(
                images,
                q,
                a_ids=a,
                image_ids=image_ids,
                region_cache_dir=region_cache_dir,
                global_cache_dir=global_cache_dir,
                save_cache=True,
            )
            loss = criterion(logits.reshape(-1, logits.size(-1)), a[:, 1:].reshape(-1))
            total_loss += float(loss.item())

        total_acc += vqa_acc(logits.argmax(dim=-1), batch["answers"], a_vocab)

    n = max(1, len(loader))
    loss_avg = total_loss / n if not greedy else 0.0
    return loss_avg, total_acc / n


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Argument haye CLI: config, resume, eval."""
    p = argparse.ArgumentParser(description="SimpleVQA — train/eval dar yek file")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--resume", default=None, help="Path be checkpoint (.pt)")
    p.add_argument(
        "--continue",
        dest="do_continue",
        action="store_true",
        help="Resume az save_dir/last.pt",
    )
    p.add_argument("--fresh", action="store_true", help="Az aval train kon (resume ignore)")
    p.add_argument("--eval", action="store_true", help="Faghat eval (greedy decode)")
    p.add_argument("--ckpt", default=None, help="Checkpoint baraye --eval")
    return p.parse_args()


def load_checkpoint(
    path: Path,
    model: VQAModel,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[StepLR],
    scaler: Optional[GradScaler],
    device: torch.device,
) -> Tuple[int, float]:
    """Checkpoint load kon; epoch va best acc ro bargardoon."""
    if not path.exists():
        print(f"Checkpoint peyda nashod: {path} — az aval shoro mikonim.")
        return 1, 0.0

    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        print(f"Format checkpoint eshtebah: {path}")
        return 1, 0.0

    unwrap_model(model).load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler"):
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception:
            pass

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best = float(ckpt.get("best", 0.0))
    print(f"Resume az {path} (next_epoch={start_epoch}, best_val_acc={best:.4f})")
    return start_epoch, best


def run_eval(cfg: Dict[str, Any], ckpt_path: str, device: torch.device) -> None:
    """Greedy decode roye val split — metric VQA v2."""
    q_vocab, a_vocab, _, val_loader = build_loaders(cfg)
    model, _ = build_vqa_model(cfg, q_vocab, a_vocab, device)

    ckpt = Path(ckpt_path).expanduser().resolve()
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state.get("model", state), strict=False)
    model.eval()

    _, acc = eval_epoch(model, val_loader, nn.CrossEntropyLoss(), a_vocab, cfg, device, greedy=True)
    print(f"Validation VQA accuracy (greedy): {acc:.4f}")


def run_train(cfg: Dict[str, Any], args: argparse.Namespace, device: torch.device) -> None:
    """Loop asli training + save checkpoint."""
    ddp_on, world, rank, local_rank = ddp_setup(cfg)

    # Finglish: seed ro per-rank shift midim ta shuffle yeksan nabashe.
    set_seed(int(cfg["seed"]) + int(rank))

    if ddp_on and device.type == "cuda":
        device = torch.device("cuda", local_rank)

    q_vocab, a_vocab, train_loader, val_loader = build_loaders(cfg)
    model, _ = build_vqa_model(cfg, q_vocab, a_vocab, device)
    if ddp_on:
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=bool(cfg.get("ddp_find_unused_parameters", False)),
        )

    # ------------------------------------------------------------------
    # Optimizer + weight decay (anti-overfit knob)
    # ------------------------------------------------------------------
    # EN: `weight_decay` adds L2 regularization to the trainable params. On the
    #     20k-qid mini set the model overfits hard (train_acc ~0.58 vs val ~0.38),
    #     so a small decay (e.g. 1e-4) pulls weights toward 0 and improves val.
    #     Default 0.0 => identical behaviour to the previous runs (reproducible).
    # FA: `weight_decay` yek regularization-e L2 roye parameter-haye trainable
    #     ezafe mikone. Roye mini set (20k) model shadidan overfit mikone, pas
    #     yek decay-e kuchik (masalan 1e-4) vazn-ha ro samt-e sefr mikeshe va
    #     val_acc ro behtar mikone. Default 0.0 => raftar-e daghighan mesl-e ghabl.
    optimizer = Adamax(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    scheduler = StepLR(
        optimizer,
        step_size=int(cfg["lr_decay_every"]),
        gamma=float(cfg["lr_decay_factor"]),
    )
    use_amp = bool(cfg["use_amp"]) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    # ------------------------------------------------------------------
    # Loss + label smoothing (anti-overfit knob)
    # ------------------------------------------------------------------
    # EN: `label_smoothing` softens the one-hot answer target so the model is not
    #     pushed to be over-confident. This directly targets the symptom where
    #     val_loss keeps rising (over-confidence) while val_acc stays flat.
    #     Default 0.0 => plain cross-entropy (previous behaviour).
    # FA: `label_smoothing` target-e one-hot ro narm mikone ta model bish-az-had
    #     motmaen nashe. Daghighan hamun moshkeli ke val_loss balatar mire vali
    #     val_acc sabet mimune ro hadaf migire. Default 0.0 => cross-entropy-e sade.
    criterion = nn.CrossEntropyLoss(
        ignore_index=0,
        label_smoothing=float(cfg.get("label_smoothing", 0.0)),
    )

    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    start_epoch = 1

    if args.fresh and args.do_continue:
        raise SystemExit("Yekish ro entekhab kon: --fresh ya --continue")

    resume_path: Optional[str] = None
    if not args.fresh:
        resume_path = args.resume
        if resume_path is None and args.do_continue:
            resume_path = str(save_dir / "last.pt")
        if resume_path is None and cfg.get("resume_from"):
            resume_path = str(cfg["resume_from"])

    if resume_path:
        start_epoch, best_acc = load_checkpoint(
            Path(resume_path).expanduser().resolve(),
            model,
            optimizer,
            scheduler,
            scaler,
            device,
        )

    epochs = int(cfg["epochs"])
    if rank == 0:
        print(
            f"device={device} ddp={ddp_on} world={world} "
            f"use_captioner={bool(cfg.get('use_captioner', True))} "
            f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
            f"q_vocab={len(q_vocab.itos)} a_vocab={len(a_vocab.itos)}"
        )

    for epoch in range(start_epoch, epochs + 1):
        # Finglish: DistributedSampler baraye shuffle bayad har epoch set_epoch beshe.
        if ddp_on:
            sampler = getattr(train_loader, "sampler", None)
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, scaler, criterion, a_vocab, cfg, device
        )
        va_loss, va_acc = eval_epoch(
            model, val_loader, criterion, a_vocab, cfg, device, greedy=False
        )
        scheduler.step()
        if rank == 0:
            print(
                f"epoch {epoch}/{epochs}  train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
                f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}"
            )

        if rank == 0:
            raw_model = unwrap_model(model)
            state = {
                "epoch": epoch,
                "best": best_acc,
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "q_vocab": q_vocab.itos,
                "a_vocab": a_vocab.itos,
                "config": cfg,
            }
            torch.save(state, save_dir / "last.pt")
            if va_acc > best_acc:
                best_acc = va_acc
                state["best"] = best_acc
                torch.save(state, save_dir / "best.pt")

    if ddp_on:
        dist.destroy_process_group()


def main() -> None:
    """Entry point: train ya eval."""
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    cfg = load_config(str(config_path))
    resolve_path_fields(cfg, PATH_KEYS)
    if isinstance(cfg.get("resume_from"), str) and cfg["resume_from"]:
        resolve_path_fields(cfg, ("resume_from",))
    set_seed(int(cfg["seed"]))

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.get("device") == "cuda" else "cpu"
    )
    print(f"config={config_path}")

    if args.eval:
        ckpt = args.ckpt or str(Path(cfg["save_dir"]) / "best.pt")
        run_eval(cfg, ckpt, device)
    else:
        run_train(cfg, args, device)


if __name__ == "__main__":
    main()
