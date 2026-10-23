"""Train QuestionDependentImageCaptioner roye QD caption JSON.

Architecture (Stage 1):
1. Faster R-CNN (freeze) → region features.
2. q_emb + GRU → qctx.
3. Yek bar question attention → attended visual.
4. concat fusion(qctx, attended) → fused.
5. LSTM: init az fused; har step attention ba h + concat(word, fused, z_t).

Run az ``QuestionDependentImageCaptioner/`` (paths relative be oon folder)::

    cd QuestionDependentImageCaptioner
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --continue
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import yaml
from PIL import Image
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW, Adamax
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decoding import generate_split_captions
from metrics import answer_hit, compute_caption_metrics
from models import build_captioner
from models.captioner_v1 import QuestionDependentImageCaptioner

"""
Finglish note (Resource optimization)
-----------------------------------
In script 3 ta feature jadid darim ke baraye Kaggle GPU limit (2xT4) kheili mohem-an:
1) AMP (mixed precision): speed/memory behtar, taghir performance kam (use_amp).
2) grad_accum_steps: batch effective ro bozorg mikone bedoon OOM.
3) region_cache: region feature FasterRCNN ro 1bar baraye har image hesab mikone va save mikone
   ta har epoch dobare detector run nashe (time saver asli).
   train/val dir joda: train_region_cache_dir vs val_region_cache_dir.
4) DDP: age ddp=true bashe, ba torchrun do ta GPU hamzaman estefade mishe.
"""


from typing import Tuple


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
    import torch.distributed as dist

    backend = str(cfg.get("ddp_backend", "nccl"))
    dist.init_process_group(backend=backend, init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True, world, rank, local_rank


def unwrap_model(model: nn.Module) -> nn.Module:
    """Finglish: DDP wrapper ro bardarim — `forward_train` faghat roye model asli hast."""
    return model.module if hasattr(model, "module") else model


TOKEN_RE = re.compile(r"[a-z0-9']+")


# ---------------------------------------------------------------------------
# Config (YAML path from CLI)
# ---------------------------------------------------------------------------
def load_config(path: str) -> Dict[str, Any]:
    """Load training hyperparameters and dataset paths from a YAML file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def print_run_config(
    cfg: Dict[str, Any],
    *,
    config_path: Optional[str] = None,
    cli: Optional[Dict[str, Any]] = None,
    save_path: Optional[Path] = None,
    rank: int = 0,
) -> None:
    """Print (and optionally persist) the full resolved training config.

    Intended for ablation / version comparison: every hyperparameter and path
    is dumped once before training so terminal logs (and ``run_config.yaml``)
    show exactly which settings produced a given run.

    Args:
        cfg: Resolved config dict (paths already absolute when applicable).
        config_path: Optional path to the YAML file that was loaded.
        cli: Optional CLI flags (e.g. ``vars(cli)``) recorded under ``_cli``.
        save_path: If set, write the same dump to this file (parent dirs created).
        rank: DDP rank; only ``rank == 0`` prints / writes.

    Note:
        Keys are sorted so two run dumps are easy to diff line-by-line.
    """
    if int(rank) != 0:
        return

    payload: Dict[str, Any] = dict(cfg)
    if config_path is not None:
        payload = {"_config_path": str(config_path), **payload}
    if cli is not None:
        payload = {**payload, "_cli": dict(cli)}

    body = yaml.safe_dump(
        payload,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=True,
    )
    banner = "=" * 72
    print(banner)
    print("RUN CONFIG (all parameters — compare versions / ablations)")
    print(banner)
    print(body.rstrip())
    print(banner)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(body, encoding="utf-8")
        print(f"run_config saved → {save_path.resolve()}")


def resolve_path_fields(cfg: Dict[str, Any], keys: Iterable[str]) -> None:
    """Expand ``~`` and resolve relative paths against the current working directory."""
    for key in keys:
        value = cfg.get(key)
        if isinstance(value, str) and value:
            cfg[key] = str(Path(value).expanduser().resolve())


def image_cap(value: Any) -> Optional[int]:
    """``null`` or non-positive -> no cap; positive int -> limit unique images."""
    if value is None:
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


# ---------------------------------------------------------------------------
# Region cache dirs (train / val joda)
# ---------------------------------------------------------------------------
def region_cache_dir_for_split(cfg: Dict[str, Any], split: str) -> Optional[str]:
    """Finglish — path cache FasterRCNN baraye train ya val.

    age ``cache_regions: false`` → None (detector har bar live).
    train → ``train_region_cache_dir`` ; val → ``val_region_cache_dir``.
    val ham mesl train read+write mikone ta epoch haye badi sari bashe.
    """
    if not bool(cfg.get("cache_regions", False)):
        return None
    if split == "test":
        # Finglish: test cache joda nadare → fallback be val cache
        return cfg.get("test_region_cache_dir") or cfg.get("val_region_cache_dir")
    key = "train_region_cache_dir" if split == "train" else "val_region_cache_dir"
    return cfg.get(key)


def snap_row_range(
    rows: List[Dict[str, Any]], row_range: Optional[List[int]]
) -> Optional[Tuple[int, int]]:
    """Finglish — [start, end) ro be marz image_id snap kon (image-disjoint split).

    Rows bayad sorted by image_id bashan (JSON QD hamintore). Har marz jolo mire
    ta yek image beyn do split nashkane; chon train.end va val.start hamoon adad
    hastan, snap yeksan mishe → split ha disjoint.
    """
    if not row_range:
        return None
    start, end = int(row_range[0]), int(row_range[1])

    def _snap(i: int) -> int:
        i = max(0, min(i, len(rows)))
        while 0 < i < len(rows) and int(rows[i]["image_id"]) == int(
            rows[i - 1]["image_id"]
        ):
            i += 1
        return i

    return _snap(start), _snap(end)


def split_rows(cfg: Dict[str, Any], split: str, qd_json: str) -> List[Dict[str, Any]]:
    """Finglish: rows yek split az ``<split>_row_range`` (age nabashe hame rows)."""
    rows = load_qd_json(qd_json)
    rng = snap_row_range(rows, cfg.get(f"{split}_row_range"))
    if rng is not None:
        rows = rows[rng[0] : rng[1]]
    return rows


def assert_disjoint_splits(cfg: Dict[str, Any]) -> None:
    """Finglish — leakage check: image_id haye train/val/test nabayad overlap dashte bashan."""
    if not cfg.get("train_row_range"):
        return
    ids = {}
    for split in ("train", "val", "test"):
        if cfg.get(f"{split}_row_range"):
            rows = split_rows(cfg, split, cfg["train_captions_json"])
            ids[split] = {int(r["image_id"]) for r in rows}
            print(f"split={split} rows={len(rows)} images={len(ids[split])}")
    names = list(ids)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            inter = ids[a] & ids[b]
            if inter:
                raise ValueError(f"Image leakage {a}/{b}: {len(inter)} shared images")


def should_run_eval(epoch: int, total_epochs: int, cfg: Dict[str, Any]) -> bool:
    """Finglish — validation har chand epoch yek bar?

    ``eval_every: 1`` → har epoch (raftar-e ghadimi).
    ``eval_every: n`` → faghat vaghti ``epoch % n == 0`` ya akharin epoch.
    Mesal: n=5, epochs=30 → val roye 5,10,15,20,25,30 (train har epoch edame dare).
    """
    every = max(1, int(cfg.get("eval_every", 1)))
    return (epoch % every == 0) or (epoch == total_epochs)


def parse_save_model_type(cfg: Dict[str, Any]) -> str:
    """Read ``save_model_type`` from config.

    Allowed values: ``epoch`` (save after each train epoch) or
    ``item`` (save every ``save_every_samples`` training samples).
    Raises ``ValueError`` for unknown types.
    """
    kind = str(cfg.get("save_model_type", "epoch")).strip().lower()
    if kind not in ("epoch", "item"):
        raise ValueError(f"save_model_type must be 'epoch' or 'item', got {kind!r}")
    return kind


def save_every_samples(cfg: Dict[str, Any]) -> int:
    """Return ``save_every_samples`` when ``save_model_type`` is ``item``.

    Must be a positive integer. Ignored when save type is ``epoch``.
    """
    n = int(cfg.get("save_every_samples", 0))
    if n <= 0:
        raise ValueError(
            "save_every_samples must be > 0 when save_model_type='item'"
        )
    return n


def init_next_save_at(samples_seen: int, every_n: int) -> int:
    """Compute the next global sample count that triggers a checkpoint save.

    First save at ``every_n``; later saves at ``2*every_n``, ``3*every_n``, ...
    Resume restores ``samples_seen`` and recomputes the next threshold.
    """
    if every_n <= 0:
        return 0
    if samples_seen <= 0:
        return every_n
    return ((samples_seen // every_n) + 1) * every_n


def global_batch_samples(batch_size: int, ddp_on: bool) -> int:
    """Global training samples in this batch (sum across DDP ranks).

    Single-process training returns ``batch_size`` unchanged.
    Under DDP, all-reduces so ``save_every_samples`` is a global count.
    """
    if batch_size <= 0 or not ddp_on:
        return batch_size
    import torch.distributed as dist

    if not dist.is_initialized():
        return batch_size
    t = torch.tensor([batch_size], dtype=torch.long)
    if torch.cuda.is_available():
        t = t.cuda()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t.item())


def build_captioner_checkpoint_state(
    model: nn.Module,
    vocab: "Vocab",
    cfg: Dict[str, Any],
    epoch: int,
    samples_seen: int,
    q_vocab: Optional["Vocab"] = None,
    best_val: float = float("inf"),
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LambdaLR] = None,
    scaler: Optional[GradScaler] = None,
) -> Dict[str, Any]:
    """Build the ``.pt`` dict written to ``last.pt`` / ``best.pt``.

    Includes model weights, caption vocabulary, optional question vocabulary
    (QD train), optimizer/scheduler/scaler, best val loss, full config, epoch
    index, and cumulative training samples.
    """
    state: Dict[str, Any] = {
        "epoch": epoch,
        "best": best_val,
        "samples_seen": samples_seen,
        "model": model_state_for_save(model, cfg),
        "vocab": vocab.itos,
        "config": cfg,
    }
    if q_vocab is not None:
        state["q_vocab"] = q_vocab.itos
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    return state


DETECTOR_PREFIX = "region_encoder.detector."


def model_state_for_save(model: nn.Module, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Finglish — ``save_detector_weights: false`` → Faster R-CNN freeze (pretrained)
    save nemishe (~160MB kamtar); load ba strict=False dobare az torchvision miad.
    """
    sd = unwrap_model(model).state_dict()
    if bool(cfg.get("save_detector_weights", True)):
        return sd
    return {k: v for k, v in sd.items() if not k.startswith(DETECTOR_PREFIX)}


def load_model_state(model: nn.Module, sd: Dict[str, Any]) -> None:
    """Finglish: state load; faghat missing key haye detector (freeze) ghabel tahamol."""
    missing, unexpected = unwrap_model(model).load_state_dict(sd, strict=False)
    bad = [k for k in missing if not k.startswith(DETECTOR_PREFIX)]
    if bad or unexpected:
        raise RuntimeError(
            f"State mismatch: missing={bad[:5]} unexpected={list(unexpected)[:5]}"
        )


def write_captioner_checkpoint(
    path: Path,
    state: Dict[str, Any],
    rank: int,
) -> None:
    """Write checkpoint file on rank 0 only (no-op on other DDP ranks)."""
    if rank != 0:
        return
    torch.save(state, path)


def maybe_save_by_samples(
    cfg: Dict[str, Any],
    model: nn.Module,
    vocab: "Vocab",
    save_dir: Path,
    epoch: int,
    samples_seen: int,
    next_save_at: int,
    rank: int,
    ddp_on: bool,
    batch_size: int,
    q_vocab: Optional["Vocab"] = None,
    best_val: float = float("inf"),
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LambdaLR] = None,
    scaler: Optional[GradScaler] = None,
) -> Tuple[int, int]:
    """Update sample counter; save ``last.pt`` when item threshold is crossed.

    Returns updated ``(samples_seen, next_save_at)``.
    No-op when ``save_model_type`` is ``epoch``.
    """
    if parse_save_model_type(cfg) != "item":
        return samples_seen, next_save_at
    every_n = save_every_samples(cfg)
    samples_seen += global_batch_samples(batch_size, ddp_on)
    while samples_seen >= next_save_at:
        state = build_captioner_checkpoint_state(
            model,
            vocab,
            cfg,
            epoch,
            samples_seen,
            q_vocab=q_vocab,
            best_val=best_val,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        write_captioner_checkpoint(save_dir / "last.pt", state, rank)
        if rank == 0:
            print(f"  checkpoint saved at samples_seen={samples_seen}")
        next_save_at += every_n
    return samples_seen, next_save_at


def tok(text: str) -> List[str]:
    """Lowercase alphanumeric tokenizer (same convention as ImageCaptioner/VQA)."""
    return TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# Vocabulary
# TODO:: in class ro matavajeh nemisham. bede AI
# ---------------------------------------------------------------------------
class Vocab:
    #
    # pad-> baraye por kardan jomalat kotah tar estefade mishe. injori lenght hame jomle ha yeksan mishe.
    # BOS-> Beginning of Sentence
    # EOS-> End of Sentence
    # UNK-> Unknown mogee ke kalme toye list ma nabashe indexesh unknows mishe.
    # min_freq-> ye tecnique hash ke baraye jelogiri az overfiting estefade mishe.
    #           kalme hayi ke kamter az min_freq toye caption ha estefade shodan ro
    #           be list vocabemon nemiyarim.
    # self.itos (Index-to-String): یک لیست که ایندکس را به کلمه نگاشت می‌کند.
    # self.stoi (String-to-Index): دیکشنری که کلمه را به ایندکس عددی تبدیل می‌کند (برای تبدیل سریع متن به عدد).

    """PAD=0, BOS=1, EOS=2, UNK=3, then frequent words."""

    PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"

    def __init__(self, words: List[str], min_freq: int = 4) -> None:
        """Build index tables from token counts in training captions."""
        counts = Counter(words)
        self.itos = [self.PAD, self.BOS, self.EOS, self.UNK] + sorted(
            w for w, n in counts.items() if n >= min_freq
        )
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, words: List[str]) -> List[int]:
        """Map tokens to ids; unknown tokens use UNK."""
        unk = self.stoi[self.UNK]
        return [self.stoi.get(w, unk) for w in words]

    @property
    def pad_id(self) -> int:
        return self.stoi[self.PAD]


# ---------------------------------------------------------------------------
# Question-dependent (QD) captions — az VQA Q+A rule-based JSON
# ---------------------------------------------------------------------------


def load_qd_json(path: str) -> List[Dict[str, Any]]:
    """QD JSON ro load kon → list sample: image_id, question, caption.

    Format (QuestionDependentCaptionGenerator):
        {"annotations": [{"image_id", "question", "caption", ...}, ...]}
    """
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data["annotations"])


def build_qd_vocabs(
    qd_json: str,
    min_freq: int,
    max_images: Optional[int] = None,
    rows: Optional[List[Dict[str, Any]]] = None,
    include_question_in_cap: bool = False,
) -> Tuple[Vocab, Vocab]:
    """Az train QD JSON do vocab besaz: caption + question (train-only, no leak).

    Args:
        qd_json: path be vqa_v2_question_dependent_captions_*.json
        min_freq: min token frequency baraye har do vocab
        max_images: age set, faghat sample haye N image_id aval
        rows: age set (train split rows), JSON dobare load nemishe
        include_question_in_cap: caption vocab = caption ∪ question words
            (copy model ha bayad har kalame soal ro betoonan copy konan)

    Returns:
        (cap_vocab, q_vocab)
    """
    if rows is None:
        rows = load_qd_json(qd_json)
    if max_images is not None and max_images > 0:
        keep_ids = set(sorted({int(r["image_id"]) for r in rows})[:max_images])
        rows = [r for r in rows if int(r["image_id"]) in keep_ids]

    cap_words: List[str] = []
    q_words: List[str] = []
    for r in rows:
        cap_words.extend(tok(str(r["caption"])))
        q_words.extend(tok(str(r["question"])))
    if include_question_in_cap:
        cap_words = cap_words + q_words
    return Vocab(cap_words, min_freq=min_freq), Vocab(q_words, min_freq=min_freq)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class VqaQdCaptionDataset(Dataset):
    """Yek sample = (image, question, QD caption) baraye question-dependent train.

    JSON az ``QuestionDependentCaptionGenerator``:
        annotations[] → image_id, question, caption
    """

    def __init__(
        self,
        images_dir: str,
        qd_json: str,
        cap_vocab: Vocab,
        q_vocab: Vocab,
        max_caption_len: int,
        max_question_len: int,
        filename_template: str,
        image_ids: Optional[List[int]] = None,
        image_size: int = 448,
        max_samples: Optional[int] = None,
        rows: Optional[List[Dict[str, Any]]] = None,
        images_from_cache: bool = False,
        region_cache_dir: Optional[str] = None,
    ) -> None:
        """QD rows ro load kon; optional filter ba image_ids / max_samples.

        Finglish — images_from_cache: age region feature har image toye
        ``region_cache_dir`` hast, JPEG load nemishe (placeholder tensor) chon
        RegionEncoder faghat az cache mikhoone. Hame image ha bayad cache shode bashan
        (``scripts/cache_regions.py``), vagarna error.
        """
        self.images_dir = Path(images_dir)
        self.cap_vocab = cap_vocab
        self.q_vocab = q_vocab
        self.max_caption_len = max_caption_len
        self.max_question_len = max_question_len
        self.filename_template = filename_template

        if rows is None:
            rows = load_qd_json(qd_json)
        if image_ids is not None:
            keep = set(int(i) for i in image_ids)
            rows = [r for r in rows if int(r["image_id"]) in keep]
        if max_samples is not None and max_samples > 0:
            rows = rows[:max_samples]

        self.samples: List[Dict[str, Any]] = [
            {
                "image_id": int(r["image_id"]),
                "question": str(r["question"]),
                "caption": str(r["caption"]),
                "answer": str(r.get("answer", "")),
                "rule": str(r.get("rule", "")),
            }
            for r in rows
        ]

        self.images_from_cache = bool(images_from_cache)
        if self.images_from_cache:
            if not region_cache_dir:
                raise ValueError("images_from_cache needs a region cache dir")
            missing = sorted(
                {
                    s["image_id"]
                    for s in self.samples
                    if not (Path(region_cache_dir) / f"{s['image_id']}.pt").exists()
                }
            )
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} images not in region cache {region_cache_dir} "
                    f"(e.g. {missing[:5]}); run scripts/cache_regions.py first"
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
        """Image + caption_ids (BOS/EOS) + question_ids (BOS/EOS) bargardoon."""
        s = self.samples[idx]
        image_id = s["image_id"]
        if self.images_from_cache:
            # Finglish: placeholder — RegionEncoder az cache mikhoone, pixel lazem nist
            image = torch.zeros(3, 1, 1)
        else:
            path = self.images_dir / self.filename_template.format(image_id=image_id)
            image = self.transform(Image.open(path).convert("RGB"))

        cap_tok = self.cap_vocab.encode(
            tok(s["caption"])[: self.max_caption_len - 2]
        )
        caption_ids = [1] + cap_tok + [2]

        q_tok = self.q_vocab.encode(
            tok(s["question"])[: self.max_question_len - 2]
        )
        question_ids = [1] + q_tok + [2]

        return {
            "image": image,
            "caption_ids": torch.tensor(caption_ids, dtype=torch.long),
            "question_ids": torch.tensor(question_ids, dtype=torch.long),
            "image_id": int(image_id),
        }


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Stack images; right-pad caption (va optional question) token ids ba PAD=0."""

    # chon hame image haro be abaad (448*448*3) dar avordim alan kafiye hamasho ba estefade az
    # stack be abaad (batch_size * 448*448*3) dar miyare.
    images = torch.stack([b["image"] for b in batch])

    # vali caption ha andaze yeksani nadaran. example: I see dog -> [100,20,50], I see dog and I run-> [100,20,50,60]
    # hala bayad har caption ha ke baad tabdil shodan be vocab niyaz be padding dadan dare ro padding bedim.

    # andaze toolani tarin batch ro peyda mikone.
    max_t = max(len(b["caption_ids"]) for b in batch)

    # ye matrix zeros be andaze max caption misazim.
    captions = torch.zeros((len(batch), max_t), dtype=torch.long)

    # alan ke in matrix zeros ro darim. bayad toye jahayi ke word darim index monaseb ro copy konim toye captions.
    # alan engar padding ro ham be caption mon emal kardim. pas hame caption ha lenght yeksan khahand dasht.
    for i, b in enumerate(batch):
        captions[i, : len(b["caption_ids"])] = b["caption_ids"]
    image_ids = torch.tensor([int(b["image_id"]) for b in batch], dtype=torch.long)

    out: Dict[str, torch.Tensor] = {
        "images": images,
        "captions": captions,
        "image_ids": image_ids,
    }

    # QD: question_ids hamishe pad mishe
    max_q = max(len(b["question_ids"]) for b in batch)
    questions = torch.zeros((len(batch), max_q), dtype=torch.long)
    for i, b in enumerate(batch):
        questions[i, : len(b["question_ids"])] = b["question_ids"]
    out["questions"] = questions

    return out


# Model dar `models/captioner_v1.py` hast (VQA ham hamoon file ro load mikone).

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def caption_token_acc(logits: torch.Tensor, targets: torch.Tensor, pad_id: int = 0) -> float:
    """Teacher-forcing token accuracy (Finglish).

    In metric chikar mikone?
        - Har step model yek vector score baraye **hame kalamat vocab** mide.
        - Ma `argmax(logits)` ro migirim → predicted next token.
        - Ba `targets` (hamoon `captions[:, 1:]`) compare mikonim.
        - PAD (``pad_id=0``) ro hesab nemikonim.

    Chera "teacher forcing"?
        - Dar train/val loss, model **kalame ghabli ground truth** ro mibine.
        - Pas in accuracy **greedy caption** ro andaze nemigire (baraye on ``eval.py`` lazeme).

    Mesal (yek caption):
        GT: "a dog on a bench"
        targets: [a, dog, on, a, bench, EOS]

        Step 1: pred=a,   target=a     → correct
        Step 2: pred=dog, target=dog   → correct
        Step 3: pred=cat, target=on    → wrong
        Step 4: pred=a,   target=a     → correct
        Step 5: pred=bench, target=bench → correct
        Step 6: pred=EOS, target=EOS   → correct

        Correct: 5
        Total positions: 6
        Accuracy for this caption: 5 / 6 = 0.833

    Args:
        logits: (batch, seq_len-1, vocab_size) — khorouji ``forward_train``.
        targets: (batch, seq_len-1) — ``captions[:, 1:]``.
        pad_id: index PAD (default 0) — inja ignore mishe.

    Returns:
        float dar baze [0, 1] — miangin accuracy roye **non-PAD** token ha.
    """
    preds = logits.argmax(dim=-1)
    mask = targets != pad_id
    n = int(mask.sum().item())
    if n == 0:
        return 0.0
    return float(((preds == targets) & mask).sum().item()) / n


# ---------------------------------------------------------------------------
# Training loop helpers
# ---------------------------------------------------------------------------
"""
    set_seed

    RNG seed-ha ra baraye reproducibility set mikonad.

    Python random, hash seed, va torch (CPU/GPU) seed
    fix mishavand ta initialization va data shuffling
    dar run-haye mokhtalef yeksan bashad.

    Input:
        seed (int)

    Output:
        None

    Note:
    Baraye reproducible research dar deep learning zaroori ast.
"""


def set_seed(seed: int) -> None:
    """Fix RNG seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def scheduled_sampling_prob(epoch: int, cfg: Dict[str, Any]) -> float:
    """Scheduled sampling probability for caption decoder (paper §5, Table 2).

    Every ``scheduled_sampling_interval`` epochs (1-indexed), add
    ``scheduled_sampling_increment`` until ``scheduled_sampling_max_prob``.
    Epochs 1..interval-1 use pure teacher forcing (p=0).
    """
    max_p = float(cfg.get("scheduled_sampling_max_prob", 0.24))
    if max_p <= 0.0:
        return 0.0
    interval = max(1, int(cfg.get("scheduled_sampling_interval", 4)))
    increment = float(cfg.get("scheduled_sampling_increment", 0.06))
    steps = max(0, epoch // interval)
    return min(max_p, steps * increment)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer, cfg: Dict[str, Any], steps_per_epoch: int
) -> LambdaLR:
    """Finglish — LR decay (paper §5, Table 2) + optional linear warmup:
        Har ``lr_decay_interval`` epoch LR × ``lr_decay_factor`` (mesal 0.6).
        Mesal LR=0.0005: epoch4→0.0003, epoch8→0.00018, epoch12→0.000108.
        ``warmup_steps`` > 0 → LR az 0 ta base ba khat (Transformer lazem dare).
        ``scheduler.step()`` ro ba'd az har optimizer step seda kon (per-step).
    """
    interval = max(1, int(cfg.get("lr_decay_interval", 4)))
    gamma = float(cfg.get("lr_decay_factor", 0.6))
    warmup = max(0, int(cfg.get("warmup_steps", 0)))
    spe = max(1, int(steps_per_epoch))

    def factor(step: int) -> float:
        # Finglish: epoch 0-index az step; decay mesl StepLR(step_size=interval)
        decay = gamma ** ((step // spe) // interval)
        warm = min(1.0, (step + 1) / warmup) if warmup > 0 else 1.0
        return decay * warm

    return LambdaLR(optimizer, lr_lambda=factor)


def build_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    """Finglish: ``optimizer: adamax`` (default, paper) ya ``adamw`` (Transformer)."""
    params = [p for p in model.parameters() if p.requires_grad]
    lr = float(cfg["learning_rate"])
    kind = str(cfg.get("optimizer", "adamax")).lower()
    if kind == "adamw":
        return AdamW(
            params,
            lr=lr,
            betas=(0.9, 0.98),
            weight_decay=float(cfg.get("weight_decay", 0.01)),
        )
    if kind == "adamax":
        return Adamax(params, lr=lr)
    raise ValueError(f"optimizer must be adamax or adamw, got {kind!r}")


"""
    train_epoch

    Yek epoch training ba cross-entropy ejra mikonad
    va weight-haye model ra update mikonad.

    Decoder input: GT previous token mixed with model prediction
    (scheduled sampling, paper §5) — probability az ``scheduled_sampling_prob``.

    Shapes:
        images: (N, C, H, W)
        captions: (N, T)
        logits: (N, T-1, V)

    Loss:
        CrossEntropy beyn
        logits.reshape(-1, V)
        va captions[:,1:].reshape(-1)

    Steps:
        forward → loss → backward → optimizer.step

    Return:
        mean cross-entropy loss (float)
"""


def train_epoch(
    model: QuestionDependentImageCaptioner,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    criterion: nn.Module,
    device: torch.device,
    cfg: Dict[str, Any],
    epoch: int,
    save_dir: Optional[Path] = None,
    vocab: Optional["Vocab"] = None,
    rank: int = 0,
    ddp_on: bool = False,
    samples_seen: int = 0,
    next_save_at: int = 0,
    q_vocab: Optional["Vocab"] = None,
    best_val: float = float("inf"),
    scheduler: Optional[LambdaLR] = None,
) -> Tuple[float, float, int, int]:
    """One training pass; returns mean loss, acc, and sample-checkpoint counters.

    When ``save_model_type=item``, saves ``last.pt`` every ``save_every_samples``
    global training samples (see ``maybe_save_by_samples``).
    batch["questions"] → ``forward_train(..., question_ids=...)``.
    """
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    accum = int(cfg.get("grad_accum_steps", 1))
    use_amp = bool(cfg.get("use_amp", False)) and device.type == "cuda"
    ss_p = scheduled_sampling_prob(epoch, cfg)
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc="train", leave=False)

    for i, batch in enumerate(pbar):
        images = batch["images"].to(device, non_blocking=device.type == "cuda")
        caps = batch["captions"].to(device, non_blocking=device.type == "cuda")
        image_ids = batch["image_ids"].to(device, non_blocking=device.type == "cuda")
        question_ids = batch["questions"].to(
            device, non_blocking=device.type == "cuda"
        )

        # dar pytorch gradient ha accumulative hastan(yani besorat default baham jaam mishan)
        # vali ma niyaz nadarim jameshon konim pas toye ebtedaye har batch gradient gabli ro none mikonim.
        # Finglish: train split → train_region_cache_dir ; miss → save (hamoon step train).
        region_cache_dir = region_cache_dir_for_split(cfg, "train")
        with autocast(enabled=use_amp):
            # feed forward mikonim ta caption ro baraye har image toye in batch peyda konim.
            logits = unwrap_model(model).forward_train(
                images,
                caps,
                question_ids=question_ids,
                image_ids=image_ids,
                region_cache_dir=region_cache_dir,
                save_region_cache=True,
                scheduled_sampling_p=ss_p,
            )

            # koss ro ba mogayese caption tolidi va ground truth hesab mikonim.
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                caps[:, 1:].reshape(-1),
            )

        scaler.scale(loss / accum).backward()
        if (i + 1) % accum == 0:
            clip = float(cfg.get("grad_clip", 0.0))
            if clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        targets = caps[:, 1:]
        total_loss += float(loss.item())
        total_acc += caption_token_acc(logits, targets)

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{caption_token_acc(logits, targets):.4f}",
        )

        if (
            save_dir is not None
            and vocab is not None
            and parse_save_model_type(cfg) == "item"
        ):
            samples_seen, next_save_at = maybe_save_by_samples(
                cfg,
                model,
                vocab,
                save_dir,
                epoch,
                samples_seen,
                next_save_at,
                rank,
                ddp_on,
                images.size(0),
                q_vocab=q_vocab,
                best_val=best_val,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
            )

    n = max(1, len(loader))
    return total_loss / n, total_acc / n, samples_seen, next_save_at


"""
    eval_epoch

    Validation epoch bedun update weight-ha.
    Gradient calculation ghayr faal ast (@no_grad).

    Shapes:
        images: (N, C, H, W)
        captions: (N, T)
        logits: (N, T-1, V)

    Loss ham mesle training (teacher forcing, scheduled_sampling_p=0)
    mohasebe mishavad ta comparable bashad.

    Return:
        mean validation cross-entropy loss (float)
"""


@torch.no_grad()
def eval_epoch(
    model: QuestionDependentImageCaptioner,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    cfg: Dict[str, Any],
    split: str = "val",
) -> Tuple[float, float]:
    """Validation loss and token accuracy (pure teacher forcing + soal)."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    # Finglish: val split → val_region_cache_dir ; miss → save (mesl train loop).
    region_cache_dir = region_cache_dir_for_split(cfg, split)
    for batch in tqdm(loader, desc="val", leave=False):
        images = batch["images"].to(device, non_blocking=device.type == "cuda")
        caps = batch["captions"].to(device, non_blocking=device.type == "cuda")
        image_ids = batch["image_ids"].to(device, non_blocking=device.type == "cuda")
        question_ids = batch["questions"].to(
            device, non_blocking=device.type == "cuda"
        )

        # inja serfan feed forward anjam midim(backward nadarim)
        logits = unwrap_model(model).forward_train(
            images,
            caps,
            question_ids=question_ids,
            image_ids=image_ids,
            region_cache_dir=region_cache_dir,
            save_region_cache=True,
            scheduled_sampling_p=0.0,
        )
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            caps[:, 1:].reshape(-1),
        )
        targets = caps[:, 1:]
        total_loss += float(loss.item())
        total_acc += caption_token_acc(logits, targets)
    n = max(1, len(loader))
    return total_loss / n, total_acc / n


# ---------------------------------------------------------------------------
# Main
    """
    parse_args

    Argument-haye CLI ra baraye training script migirad.
    Dar in project faghat path file config (YAML) az command
    line gerefte mishavad.

    Config file tamami hyperparameter-ha ra dar khod darad
    (mesl batch_size, learning_rate, dataset paths).

    Input:
        CLI argument:
            --config : path be YAML config file

    Output:
        argparse.Namespace

    Example:
        python train.py --config configs/default.yaml

    Note:
        YAML config badan baraye sakhte dataset, model
        va training settings estefade mishavad.
    """
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """CLI: path to YAML config (all hyperparameters live in the config file)."""
    p = argparse.ArgumentParser(description="Train QuestionDependentImageCaptioner")
    p.add_argument(
        "--config",
        default="configs/default.yaml",
        help="YAML config (relative to project folder unless absolute)",
    )
    p.add_argument("--resume", default=None, help="Path to checkpoint (.pt)")
    p.add_argument("--seed", type=int, default=None, help="Override cfg seed")
    p.add_argument("--save-dir", default=None, help="Override cfg save_dir")
    p.add_argument(
        "--continue",
        dest="do_continue",
        action="store_true",
        help="Resume from save_dir/last.pt",
    )
    p.add_argument(
        "--fresh",
        action="store_true",
        help="Train from scratch (ignore resume)",
    )
    return p.parse_args()


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[LambdaLR],
    scaler: Optional[GradScaler],
    device: torch.device,
) -> Tuple[int, float, int]:
    """Load checkpoint; return (start_epoch, best_val_loss, samples_seen)."""
    if not path.exists():
        print(f"Checkpoint not found: {path} — starting from scratch.")
        return 1, float("inf"), 0

    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        print(f"Invalid checkpoint format: {path}")
        return 1, float("inf"), 0

    load_model_state(model, ckpt["model"])
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
    best_val = float(ckpt.get("best", float("inf")))
    samples_seen = int(ckpt.get("samples_seen", 0))
    print(
        f"Resume from {path} (next_epoch={start_epoch}, "
        f"best_val_loss={best_val:.4f}, samples_seen={samples_seen})"
    )
    return start_epoch, best_val, samples_seen

#
    """
    main

    Pipeline asli training ra ejra mikonad:
    config → dataset → dataloader → model → training loop.

    Steps:
    1) load config az YAML
    2) build vocabulary az captions
    3) sakhte VqaQdCaptionDataset
    4) sakhte DataLoader baraye train/val
    5) initialize QuestionDependentImageCaptioner
    6) run train_epoch va eval_epoch
    7) save last.pt va best.pt checkpoints

    Important Tensor Shapes:
        images: (N, C, H, W)
        captions: (N, T)
        logits: (N, T-1, vocab_size)

    Example:
        logits[3,5] → score tamami kalamat vocab
        baraye kalame 6om caption image 4.

    Note:
        validation loss baraye entekhab best model
        estefade mishavad.
    """
#


def build_split_dataset(
    cfg: Dict[str, Any],
    split: str,
    vocab: Vocab,
    q_vocab: Vocab,
) -> VqaQdCaptionDataset:
    """Finglish — dataset yek split (train/val/test).

    Age ``<split>_row_range`` set bashe (benchmark): rows az ``train_captions_json``
    ba marz image snap mishe. Vagarna raftar ghadimi (max_*_images / max_*_samples).
    """
    if cfg.get(f"{split}_row_range"):
        qd_json = cfg["train_captions_json"]
        rows = split_rows(cfg, split, qd_json)
        image_ids, max_samples = None, None
    else:
        key = "train" if split == "train" else "val"
        qd_json = cfg[f"{key}_captions_json"]
        rows = None
        max_images = image_cap(cfg.get(f"max_{key}_images"))
        max_samples = image_cap(cfg.get(f"max_{key}_samples"))
        image_ids = None
        if max_images is not None:
            image_ids = sorted(
                {int(r["image_id"]) for r in load_qd_json(qd_json)}
            )[:max_images]
    img_key = "train" if split == "train" else "val"
    return VqaQdCaptionDataset(
        cfg[f"{img_key}_images_dir"],
        qd_json,
        vocab,
        q_vocab,
        int(cfg["max_caption_len"]),
        int(cfg.get("max_question_len", 14)),
        cfg[f"{img_key}_image_filename_template"],
        image_ids=image_ids,
        image_size=int(cfg.get("image_size", 448)),
        max_samples=max_samples,
        rows=rows,
        images_from_cache=bool(cfg.get("images_from_cache", False)),
        region_cache_dir=region_cache_dir_for_split(cfg, split),
    )


def build_vocabs_from_cfg(cfg: Dict[str, Any]) -> Tuple[Vocab, Vocab]:
    """Finglish: vocab faghat az train split (benchmark) ya kol train JSON (ghadimi)."""
    rows = None
    if cfg.get("train_row_range"):
        rows = split_rows(cfg, "train", cfg["train_captions_json"])
    return build_qd_vocabs(
        cfg["train_captions_json"],
        int(cfg["vocab_min_freq"]),
        image_cap(cfg.get("max_train_images")),
        rows=rows,
        include_question_in_cap=bool(cfg.get("cap_vocab_include_question", False)),
    )


def val_caption_scores(
    model: nn.Module,
    loader: DataLoader,
    vocab: Vocab,
    cfg: Dict[str, Any],
    device: torch.device,
    split: str = "val",
) -> Dict[str, float]:
    """Finglish — greedy (beam=1) decode roye val → CIDEr/BLEU/ROUGE + AnswerHit."""
    hyps, _ = generate_split_captions(
        unwrap_model(model),
        loader,
        vocab.itos,
        device,
        int(cfg["max_caption_len"]),
        beam_size=int(cfg.get("val_beam_size", 1)),
        region_cache_dir=region_cache_dir_for_split(cfg, split),
    )
    samples = loader.dataset.samples
    refs = [[tok(s["caption"])] for s in samples]
    scores = compute_caption_metrics(hyps, refs)
    scores.update(answer_hit(hyps, [tok(s["answer"]) for s in samples]))
    return scores


def selection_value(
    select_metric: str, va_loss: float, scores: Optional[Dict[str, float]]
) -> float:
    """Finglish: 'lower is better' value baraye best.pt (loss ya -CIDEr)."""
    if select_metric == "cider":
        assert scores is not None
        return -float(scores["CIDEr"])
    return float(va_loss)


def main() -> None:
    """Load config, build QD data + captioner (registry), train/val."""
    cli = parse_args()
    config_path = Path(cli.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    cfg = load_config(str(config_path))
    # Finglish: CLI override baraye multi-seed benchmark (yek YAML, chand seed)
    if cli.seed is not None:
        cfg["seed"] = int(cli.seed)
    if cli.save_dir is not None:
        cfg["save_dir"] = cli.save_dir
    resolve_path_fields(
        cfg,
        (
            "train_captions_json",
            "val_captions_json",
            "train_images_dir",
            "val_images_dir",
            "save_dir",
            "train_region_cache_dir",
            "val_region_cache_dir",
            "test_region_cache_dir",
        ),
    )
    if isinstance(cfg.get("resume_from"), str) and cfg["resume_from"]:
        resolve_path_fields(cfg, ("resume_from",))
    ddp_on, world, rank, local_rank = ddp_setup(cfg)

    # Finglish: baraye DDP, seed ro ham per-rank shift midim ta shuffle yeksan nabashe.
    set_seed(int(cfg["seed"]) + int(rank))

    print_run_config(
        cfg,
        config_path=str(config_path),
        cli=vars(cli),
        save_path=Path(cfg["save_dir"]) / "run_config.yaml",
        rank=rank,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.get("device") == "cuda" else "cpu"
    )
    if ddp_on and device.type == "cuda":
        device = torch.device("cuda", local_rank)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    assert_disjoint_splits(cfg)

    # Finglish — QD: do vocab (caption + question) az train split
    vocab, q_vocab = build_vocabs_from_cfg(cfg)
    train_ds = build_split_dataset(cfg, "train", vocab, q_vocab)
    val_ds = build_split_dataset(cfg, "val", vocab, q_vocab)

    device_is_cuda = device.type == "cuda"
    loader_kw = {
        "batch_size": int(cfg["batch_size"]),
        "num_workers": int(cfg["num_workers"]),
        "collate_fn": collate_batch,
        "pin_memory": bool(cfg.get("pin_memory", False)) and device_is_cuda,
    }
    if int(cfg["num_workers"]) > 0:
        loader_kw["persistent_workers"] = bool(cfg.get("persistent_workers", False))
        loader_kw["prefetch_factor"] = int(cfg.get("prefetch_factor", 2))
    train_sampler = DistributedSampler(train_ds, shuffle=True) if ddp_on else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if ddp_on else None
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

    model = build_captioner(cfg, vocab, q_vocab).to(device)
    if ddp_on:
        # Finglish: DDP baraye 2xT4 Kaggle. torchrun required.
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=bool(cfg.get("ddp_find_unused_parameters", False)),
        )

    accum = max(1, int(cfg.get("grad_accum_steps", 1)))
    optimizer = build_optimizer(model, cfg)
    scheduler = build_lr_scheduler(
        optimizer, cfg, steps_per_epoch=math.ceil(len(train_loader) / accum)
    )
    # Finglish: label smoothing (bench=0.1). Copy model ha log-prob midan;
    # log_softmax(log p) = log p → CrossEntropy hamoon NLL mishe.
    criterion = nn.CrossEntropyLoss(
        ignore_index=0, label_smoothing=float(cfg.get("label_smoothing", 0.0))
    )
    eval_criterion = nn.CrossEntropyLoss(ignore_index=0)
    use_amp = bool(cfg.get("use_amp", False)) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    start_epoch = 1
    epochs = int(cfg["epochs"])
    save_type = parse_save_model_type(cfg)
    save_last = bool(cfg.get("save_last", True))
    select_metric = str(cfg.get("select_metric", "loss")).lower()
    if select_metric not in ("loss", "cider"):
        raise ValueError(f"select_metric must be loss or cider, got {select_metric!r}")
    samples_seen = 0

    if cli.fresh and cli.do_continue:
        raise SystemExit("Choose one: --fresh or --continue")

    resume_path: Optional[str] = None
    if not cli.fresh:
        resume_path = cli.resume
        if resume_path is None and cli.do_continue:
            resume_path = str(save_dir / "last.pt")
        if resume_path is None and cfg.get("resume_from"):
            resume_path = str(cfg["resume_from"])

    if resume_path:
        start_epoch, best_val, samples_seen = load_checkpoint(
            Path(resume_path).expanduser().resolve(),
            model,
            optimizer,
            scheduler,
            scaler,
            device,
        )

    next_save_at = (
        init_next_save_at(samples_seen, save_every_samples(cfg))
        if save_type == "item"
        else 0
    )

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        print(f"config={config_path}")
        print(
            f"device={device} ddp={ddp_on} world={world} dataset=qd "
            f"model_class={cfg.get('model_class', 'QuestionDependentImageCaptioner')} "
            f"trainable_params={n_trainable:,} "
            f"train_rows={len(train_ds)} val_rows={len(val_ds)} "
            f"vocab={len(vocab.itos)} q_vocab={len(q_vocab.itos)} "
            f"save_model_type={save_type} select_metric={select_metric}"
        )
        if save_type == "item":
            print(f"save_every_samples={save_every_samples(cfg)}")

    history: List[Dict[str, Any]] = []
    for epoch in range(start_epoch, epochs + 1):
        if ddp_on and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        ss_p = scheduled_sampling_prob(epoch, cfg)
        t_epoch = time.perf_counter()
        tr_loss, tr_acc, samples_seen, next_save_at = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            criterion,
            device,
            cfg,
            epoch,
            save_dir=save_dir,
            vocab=vocab,
            rank=rank,
            ddp_on=ddp_on,
            samples_seen=samples_seen,
            next_save_at=next_save_at,
            q_vocab=q_vocab,
            best_val=best_val,
            scheduler=scheduler,
        )
        train_secs = time.perf_counter() - t_epoch
        # Finglish: eval har n epoch — na har dafe (eval_every az config).
        run_eval = should_run_eval(epoch, epochs, cfg)
        scores: Optional[Dict[str, float]] = None
        if run_eval:
            va_loss, va_acc = eval_epoch(
                model, val_loader, eval_criterion, device, cfg
            )
            if select_metric == "cider":
                scores = val_caption_scores(model, val_loader, vocab, cfg, device)
        if rank == 0:
            cur_lr = optimizer.param_groups[0]["lr"]
            msg = (
                f"epoch {epoch}/{epochs}  ss_p={ss_p:.2f}  lr={cur_lr:.2e}  "
                f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
                f"({train_secs:.1f}s)"
            )
            if run_eval:
                msg += f"  val_loss={va_loss:.4f} val_acc={va_acc:.4f}"
                if scores is not None:
                    msg += (
                        f" val_CIDEr={scores['CIDEr']:.4f}"
                        f" val_BLEU-4={scores['BLEU-4']:.4f}"
                        f" val_AnswerHit={scores['AnswerHit']:.4f}"
                    )
            else:
                msg += f"  val=skip (eval_every={max(1, int(cfg.get('eval_every', 1)))})"
            print(msg)
            row: Dict[str, Any] = {
                "epoch": epoch,
                "lr": cur_lr,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "train_secs": train_secs,
            }
            if run_eval:
                row.update({"val_loss": va_loss, "val_acc": va_acc})
                if scores is not None:
                    row.update({f"val_{k}": v for k, v in scores.items()})
            history.append(row)
            (save_dir / "history.json").write_text(
                json.dumps(history, indent=1), encoding="utf-8"
            )

        if rank == 0:
            state = build_captioner_checkpoint_state(
                model,
                vocab,
                cfg,
                epoch,
                samples_seen,
                q_vocab=q_vocab,
                best_val=best_val,
                optimizer=optimizer if save_last else None,
                scheduler=scheduler if save_last else None,
                scaler=scaler if save_last else None,
            )
            if save_type == "epoch" and save_last:
                write_captioner_checkpoint(save_dir / "last.pt", state, rank)
            if run_eval:
                value = selection_value(select_metric, va_loss, scores)
                if value < best_val:
                    best_val = value
                    state["best"] = best_val
                    state["select_metric"] = select_metric
                    state["best_epoch"] = epoch
                    # Finglish: best.pt baraye benchmark — optimizer state lazem nist
                    for k in ("optimizer", "scheduler", "scaler"):
                        state.pop(k, None)
                    write_captioner_checkpoint(save_dir / "best.pt", state, rank)
                    print(f"  best.pt updated (epoch {epoch}, {select_metric})")

    if rank == 0 and save_type == "item":
        state = build_captioner_checkpoint_state(
            model,
            vocab,
            cfg,
            epochs,
            samples_seen,
            q_vocab=q_vocab,
            best_val=best_val,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        write_captioner_checkpoint(save_dir / "last.pt", state, rank)

    if ddp_on:
        import torch.distributed as dist

        dist.destroy_process_group()


if __name__ == "__main__":
    main()
