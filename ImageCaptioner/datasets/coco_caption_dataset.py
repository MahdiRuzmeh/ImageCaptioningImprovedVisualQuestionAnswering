"""MSCOCO caption dataset — stage-one training for ``ImageCaptionerV1``.

Thesis / paper positioning (*Image captioning improved visual question answering*)
-------------------------------------------------------------------------------
This module implements **supervised image captioning only** (images + reference captions).
The captioner is later frozen inside ``VQAModel`` so its sentence embedding supplies auxiliary
semantics for VQA — cite the thesis subsection that introduces the **two-stage** pipeline:
**(A)** caption pre-training, **(B)** VQA fine-tuning / joint inference with frozen caption weights.

Layout (config-driven)
----------------------
Paths come from YAML (``train_captions_json``, ``val_captions_json``, ``train_images_dir``,
``val_images_dir``, filename templates). Official MSCOCO **train** and **validation** JSON + image
folders are expected; no random 80/20 split over one split.

Each JSON follows MSCOCO caption format: ``annotations[].image_id`` and ``annotations[].caption``.
One training row is created **per caption sentence**, so popular images appear multiple times.

Interaction with VQA
--------------------
Tokenizer ``tok`` matches ``VQA.datasets.vqa_dataset.tok`` so word statistics stay comparable.
Training calls ``forward_train(images, caption_ids)`` without question ids by default; optional
question-conditioned captioning is described in ``ImageCaptionerV1`` docstrings.

Examples
--------
Load captions and build datasets (paths from config)::

    from datasets.coco_caption_dataset import load_caps, build_vocab, CocoCaptionDataset

    caps = load_caps("/path/to/captions_train2014.json")
    vocab = build_vocab("/path/to/captions_train2014.json", min_freq=4)
    ds = CocoCaptionDataset(
        images_dir="/path/to/train2014",
        captions_json="/path/to/captions_train2014.json",
        vocab=vocab,
        max_len=20,
        image_filename_template="COCO_train2014_{image_id:012d}.jpg",
    )

Inspect one sample::

    x = ds[0]
    # x["image"] shape: (3, 448, 448)
    # x["caption_ids"] starts with BOS id 1, ends with EOS id 2 (same convention as VQA q/a)
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

TOKEN_RE = re.compile(r"[a-z0-9']+")


def tok(text: str) -> List[str]:
    """Shared tokenizer with VQA (lowercase/alphanumeric).

    Examples:
        >>> tok("A dog runs.")
        ['a', 'dog', 'runs']
    """
    return TOKEN_RE.findall(text.lower())


def load_caps(captions_json: str) -> Dict[int, List[str]]:
    """Load a MSCOCO captions JSON into ``image_id -> [caption str, ...]``.

    Args:
        captions_json: Path to ``captions_train2014.json`` or ``captions_val2014.json``.
    """
    with Path(captions_json).open("r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[int, List[str]] = {}
    for ann in data["annotations"]:
        out.setdefault(int(ann["image_id"]), []).append(ann["caption"])
    return out


class Vocab:
    """Caption vocabulary; specials identical to VQA for embedding compatibility.

    Examples:
        >>> v = Vocab(["sky", "sky", "boat"], min_freq=2)
        >>> "<pad>" in v.itos
        True
    """

    PAD = "<pad>"
    BOS = "<bos>"
    EOS = "<eos>"
    UNK = "<unk>"

    def __init__(self, words: List[str], min_freq: int = 4) -> None:
        """Build ``itos``/``stoi`` from token counts (PAD/BOS/EOS/UNK first, then frequent words).

        Args:
            words: Flat list of caption tokens (e.g. from ``tok`` over training captions).
            min_freq: Minimum count for a word to be included; rarer tokens map to UNK.
        """
        from collections import Counter

        c = Counter(words)
        self.itos = [self.PAD, self.BOS, self.EOS, self.UNK] + sorted([w for w, n in c.items() if n >= min_freq])
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, words: List[str]) -> List[int]:
        """Map tokens to indices; unknown words use the UNK index."""
        return [self.stoi.get(w, self.stoi[self.UNK]) for w in words]

    @property
    def pad_id(self) -> int:
        """Padding index (0) for labels and batch padding, aligned with VQA vocabs."""
        return self.stoi[self.PAD]


def build_vocab(captions_json: str, min_freq: int) -> Vocab:
    """Collect tokens from all captions in the **training** captions JSON."""
    caps = load_caps(captions_json)
    words: List[str] = []
    for i in caps:
        for cap in caps.get(i, []):
            words.extend(tok(cap))
    return Vocab(words, min_freq=min_freq)


class CocoCaptionDataset(Dataset):
    """Expand ``(image_id, caption string)`` pairs with torchvision preprocessing.

    Images are resized to ``448×448`` and normalized with ImageNet statistics — consistent with
    ``VQADataset`` so the captioner accepts tensors identical to those passed into ``VQAModel``.

    Examples:
        Training loop snippet::

            for batch in loader:
                logits = model.forward_train(batch["images"], batch["captions"])
                # caption tokens align with logits[:, t, :] predicting caption[:, t+1]
    """

    def __init__(
        self,
        images_dir: str,
        captions_json: str,
        vocab: Vocab,
        max_len: int = 20,
        image_filename_template: str = "COCO_train2014_{image_id:012d}.jpg",
        image_ids: Optional[List[int]] = None,
    ) -> None:
        """Index captions from ``captions_json`` under ``images_dir``.

        Args:
            images_dir: Folder containing COCO JPEGs (e.g. ``train2014`` or ``val2014``).
            captions_json: MSCOCO captions file for this split.
            vocab: Caption vocabulary (specials aligned with VQA).
            max_len: Max caption length in tokens including BOS/EOS framing.
            image_filename_template: ``str.format`` pattern with ``image_id`` (MSCOCO id integer).
            image_ids: Optional subset of image ids; default is every id present in the JSON.
        """
        self.images_dir = Path(images_dir)
        self.vocab = vocab
        self.max_len = max_len
        self.image_filename_template = image_filename_template
        self.samples: List[Tuple[int, str]] = []
        caps = load_caps(captions_json)
        ids = sorted(image_ids) if image_ids is not None else sorted(caps.keys())
        for i in ids:
            for c in caps.get(i, []):
                self.samples.append((i, c))
        self.tf = transforms.Compose(
            [
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        """Number of (image, caption) rows (one per caption sentence, not per unique image)."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load image tensor and BOS/EOS-framed caption token ids up to ``max_len``.

        Returns:
            Dict with ``image`` (CHW float) and ``caption_ids`` (1D long tensor).
        """
        image_id, cap = self.samples[idx]
        name = self.image_filename_template.format(image_id=image_id)
        image_path = self.images_dir / name
        image = self.tf(Image.open(image_path).convert("RGB"))
        ids = [1] + self.vocab.encode(tok(cap)[: self.max_len - 2]) + [2]
        return {"image": image, "caption_ids": torch.tensor(ids, dtype=torch.long)}


def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Right-pad ``caption_ids`` with zeros for batching.

    Examples:
        >>> # dl = DataLoader(ds, batch_size=16, collate_fn=collate)
    """
    images = torch.stack([b["image"] for b in batch])
    m = max(len(b["caption_ids"]) for b in batch)
    caps = torch.zeros((len(batch), m), dtype=torch.long)
    for i, b in enumerate(batch):
        caps[i, : len(b["caption_ids"])] = b["caption_ids"]
    return {"images": images, "captions": caps}
