"""Train a minimal region-attention LSTM image captioner (one file).

Paper reference (*Image captioning improved visual question answering*, §3.3)
---------------------------------------------------------------------------
1. Encoder: K region vectors ``v_i`` from Faster R-CNN (frozen).
2. Decoder: at each step ``t``, attention weights ``α_{ti} ∝ exp(f_att(v_i, h_{t-1}))``,
   context ``z_t = Σ_i α_{ti} v_i``, then LSTM predicts the next word from
   ``[word_embedding ; z_t]``.

Paper sizes (§3.1, §5, Table 2): regions ``v_i ∈ ℝ^{2048}``, LSTM / word / attention
working dim **512**, **32** regions. Project ``h_{t-1}`` and each ``v_i`` to 512 for scores,
sum weighted **2048-D** regions, then project context to 512 for the LSTM.

Run from ``SimpleImageCaptioner/`` (paths in YAML are relative to that folder)::

    cd SimpleImageCaptioner
    python train.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import yaml
from PIL import Image
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.captioner_v1 import SimpleImageCaptioner

TOKEN_RE = re.compile(r"[a-z0-9']+")


# ---------------------------------------------------------------------------
# Config (YAML path from CLI)
# ---------------------------------------------------------------------------
def load_config(path: str) -> Dict[str, Any]:
    """Load training hyperparameters and dataset paths from a YAML file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


# in function karesh ine file caption MSCOCO ro be ye dictionary tabdil kone.
# {image_id1 => [caption1,caption2, ...], ...}
def load_caps_json(path: str) -> Dict[int, List[str]]:
    """MSCOCO captions JSON -> ``{image_id: [caption, ...]}``."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[int, List[str]] = {}
    for ann in data["annotations"]:
        out.setdefault(int(ann["image_id"]), []).append(ann["caption"])
    return out

# in function karesh ine chand caption/hame caption haro dar miyare. [c1,c2,c3, ...]
# badan miyad ye object az class Vocab misaze va caption haro behesh pass mide.


def build_vocab(captions_json: str, min_freq: int, max_images: Optional[int]) -> Vocab:
    """Collect tokens from training captions (optionally capped image count)."""
    caps = load_caps_json(captions_json)
    ids = sorted(caps.keys())
    if max_images and max_images > 0:
        ids = ids[:max_images]

    # in words dar higigat listi az caption ha ast. yani: [c1, c2,c3, ...]
    words: List[str] = []
    for i in ids:
        for c in caps[i]:
            words.extend(tok(c))
    return Vocab(words, min_freq=min_freq)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CocoCaptionDataset(Dataset):
    """One row per (image, caption sentence) with 448×448 ImageNet normalization."""

    def __init__(
        self,
        images_dir: str,
        captions_json: str,
        vocab: Vocab,
        # max lenght caption
        max_len: int,
        filename_template: str,
        image_ids: Optional[List[int]] = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.vocab = vocab
        self.max_len = max_len
        self.filename_template = filename_template
        self.samples: List[Tuple[int, str]] = []

        # listi az dictionary has:
        # [[image_id_1,[c1,c2,c3]],[image_id_2,[c4,c5,c6]], ...]
        caps = load_caps_json(captions_json)

        ids = sorted(image_ids) if image_ids else sorted(caps.keys())
        for i in ids:
            for c in caps.get(i, []):
                # toye samples miyaym caps ro flatten mikonim. engar har be ezaye har image va caption ye item inja misazim.
                # yani [[image_id1,caption1],[image_id1,caption2],[image_id1,caption3], ..., [image_id2,caption6]]
                self.samples.append((i, c))

        # TODO:: in chiye?
        # engar image haro be size (448,448) tabdil mikone. bagiyasho nemidonam.
        self.transform = transforms.Compose(
            [
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    # in function image id migire va ye dictionary barmigardone: {image, caption_ids}
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_id, caption = self.samples[idx]
        path = self.images_dir / \
            self.filename_template.format(image_id=image_id)

        # size image ro be chizi ke mikhaym tabdil mikonim
        image = self.transform(Image.open(path).convert("RGB"))

        # tok(caption)[: self.max_len - 2]:
        #  ما متن را توکنایز می‌کنیم و طول آن را محدود می‌کنیم. چرا -2؟
        # چون قرار است دو توکن ویژه به ابتدا و انتهای آن اضافه کنیم (<bos> و <eos>).
        tokens = self.vocab.encode(tok(caption)[: self.max_len - 2])

        # [1]=> BOS, [2]=> EOS
        caption_ids = [1] + tokens + [2]
        return {"image": image, "caption_ids": torch.tensor(caption_ids, dtype=torch.long)}

# TODO:: inam bede AI bebinam chikar mikone.


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Stack images; right-pad caption token ids with PAD (0)."""

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
    return {"images": images, "captions": captions}


# Model dar `models/captioner_v1.py` hast (VQA ham hamoon file ro load mikone).

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


"""
    train_epoch

    Yek epoch training ba teacher forcing ejra mikonad
    va weight-haye model ra update mikonad.

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
    model: SimpleImageCaptioner,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """One training pass; returns mean cross-entropy loss."""
    model.train()
    total = 0.0

    # ba class tqdm progress bar baraye training neshon midim
    pbar = tqdm(loader, desc="train", leave=False)

    for batch in pbar:
        images = batch["images"].to(device)
        caps = batch["captions"].to(device)

        # dar pytorch gradient ha accumulative hastan(yani besorat default baham jaam mishan)
        # vali ma niyaz nadarim jameshon konim pas toye ebtedaye har batch gradient gabli ro none mikonim.
        optimizer.zero_grad(set_to_none=True)

        # feed forward mikonim ta caption ro baraye har image toye in batch peyda konim.
        logits = model.forward_train(images, caps)

        # koss ro ba mogayese caption tolidi va ground truth hesab mikonim.
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            caps[:, 1:].reshape(-1),
        )

        # backward pass ro barmigardim ta weight haro update konim.
        loss.backward()
        optimizer.step()

        # loss kole batch haye in epoch ro hesab mikonim.
        total += float(loss.item())

        # loss in batch ro toye terminal neshon midim.
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total / max(1, len(loader))


"""
    eval_epoch

    Validation epoch bedun update weight-ha.
    Gradient calculation ghayr faal ast (@no_grad).

    Shapes:
        images: (N, C, H, W)
        captions: (N, T)
        logits: (N, T-1, V)

    Loss ham mesle training (teacher forcing)
    mohasebe mishavad ta comparable bashad.

    Return:
        mean validation cross-entropy loss (float)
"""


@torch.no_grad()
def eval_epoch(
    model: SimpleImageCaptioner,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Validation loss (teacher forcing, same as training)."""
    model.eval()
    total = 0.0
    for batch in tqdm(loader, desc="val", leave=False):
        images = batch["images"].to(device)
        caps = batch["captions"].to(device)

        # inja serfan feed forward anjam midim(backward nadarim)
        logits = model.forward_train(images, caps)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            caps[:, 1:].reshape(-1),
        )
        total += float(loss.item())
    return total / max(1, len(loader))


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
    p = argparse.ArgumentParser(description="Train SimpleImageCaptioner")
    p.add_argument(
        "--config",
        default="configs/default.yaml",
        help="YAML config (relative to SimpleImageCaptioner/ unless absolute)",
    )
    return p.parse_args()

#
    """
    main

    Pipeline asli training ra ejra mikonad:
    config → dataset → dataloader → model → training loop.

    Steps:
    1) load config az YAML
    2) build vocabulary az captions
    3) sakhte CocoCaptionDataset
    4) sakhte DataLoader baraye train/val
    5) initialize SimpleImageCaptioner
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


def main() -> None:
    """Load config from CLI, then build data, model, and run train/val loops."""
    cli = parse_args()
    config_path = Path(cli.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    cfg = load_config(str(config_path))
    resolve_path_fields(
        cfg,
        (
            "train_captions_json",
            "val_captions_json",
            "train_images_dir",
            "val_images_dir",
            "save_dir",
        ),
    )
    set_seed(int(cfg["seed"]))

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.get("device") == "cuda" else "cpu"
    )
    max_train = image_cap(cfg.get("max_train_images"))
    max_val = image_cap(cfg.get("max_val_images"))
    train_ids = None
    val_ids = None
    if max_train is not None:
        train_ids = sorted(load_caps_json(
            cfg["train_captions_json"]).keys())[:max_train]
    if max_val is not None:
        val_ids = sorted(load_caps_json(
            cfg["val_captions_json"]).keys())[:max_val]

    vocab = build_vocab(
        cfg["train_captions_json"],
        int(cfg["vocab_min_freq"]),
        max_train,
    )

    train_ds = CocoCaptionDataset(
        cfg["train_images_dir"],
        cfg["train_captions_json"],
        vocab,
        int(cfg["max_caption_len"]),
        cfg["train_image_filename_template"],
        image_ids=train_ids,
    )
    val_ds = CocoCaptionDataset(
        cfg["val_images_dir"],
        cfg["val_captions_json"],
        vocab,
        int(cfg["max_caption_len"]),
        cfg["val_image_filename_template"],
        image_ids=val_ids,
    )

    loader_kw = {
        "batch_size": int(cfg["batch_size"]),
        "num_workers": int(cfg["num_workers"]),
        "collate_fn": collate_batch,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

    hidden_dim = int(cfg.get("hidden_dim", cfg.get("lstm_hidden", 512)))
    model = SimpleImageCaptioner(
        vocab_size=len(vocab.itos),
        pad_id=vocab.pad_id,
        word_dim=int(cfg["word_dim"]),
        hidden_dim=hidden_dim,
        max_regions=int(cfg["max_regions"]),
        question_dim=int(cfg.get("question_dim", cfg["word_dim"])),
        embed_dim=int(cfg.get("embed_dim", hidden_dim)),
        region_dim=int(cfg["region_dim"]),
    ).to(device)

    #TODO:: Adam -> adagrad
    optimizer = Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    epochs = int(cfg["epochs"])

    print(f"config={config_path}")
    print(
        f"device={device} train_rows={len(train_ds)} val_rows={len(val_ds)} "
        f"vocab={len(vocab.itos)}"
    )

    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch(model, train_loader,
                              optimizer, criterion, device)
        va_loss = eval_epoch(model, val_loader, criterion, device)
        print(
            f"epoch {epoch}/{epochs}  train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}")

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "vocab": vocab.itos,
            "config": cfg,
        }
        torch.save(state, save_dir / "last.pt")
        if va_loss < best_val:
            best_val = va_loss
            torch.save(state, save_dir / "best.pt")


if __name__ == "__main__":
    main()
