import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

TOKEN_RE = re.compile(r"[a-z0-9']+")


def tok(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def load_caps(dataset_root: str) -> Dict[int, List[str]]:
    with Path(dataset_root, "captions_val2014.json").open("r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[int, List[str]] = {}
    for ann in data["annotations"]:
        out.setdefault(int(ann["image_id"]), []).append(ann["caption"])
    return out


def split_ids(ids: List[int], seed: int = 42) -> Tuple[List[int], List[int]]:
    ids = sorted(set(ids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = int(0.8 * len(ids))
    return ids[:n], ids[n:]


class Vocab:
    PAD = "<pad>"
    BOS = "<bos>"
    EOS = "<eos>"
    UNK = "<unk>"

    def __init__(self, words: List[str], min_freq: int = 4) -> None:
        from collections import Counter
        c = Counter(words)
        self.itos = [self.PAD, self.BOS, self.EOS, self.UNK] + sorted([w for w, n in c.items() if n >= min_freq])
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, words: List[str]) -> List[int]:
        return [self.stoi.get(w, self.stoi[self.UNK]) for w in words]

    @property
    def pad_id(self) -> int:
        return self.stoi[self.PAD]


def build_vocab(dataset_root: str, train_ids: List[int], min_freq: int) -> Vocab:
    caps = load_caps(dataset_root)
    words: List[str] = []
    for i in train_ids:
        for cap in caps.get(i, []):
            words.extend(tok(cap))
    return Vocab(words, min_freq=min_freq)


class CocoCaptionDataset(Dataset):
    def __init__(self, dataset_root: str, image_ids: List[int], vocab: Vocab, max_len: int = 20) -> None:
        self.root = Path(dataset_root)
        self.vocab = vocab
        self.max_len = max_len
        self.samples: List[Tuple[int, str]] = []
        caps = load_caps(dataset_root)
        for i in image_ids:
            for c in caps.get(i, []):
                self.samples.append((i, c))
        self.tf = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_id, cap = self.samples[idx]
        image_path = self.root / "val2014" / f"COCO_val2014_{image_id:012d}.jpg"
        image = self.tf(Image.open(image_path).convert("RGB"))
        ids = [1] + self.vocab.encode(tok(cap)[: self.max_len - 2]) + [2]
        return {"image": image, "caption_ids": torch.tensor(ids, dtype=torch.long)}


def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    images = torch.stack([b["image"] for b in batch])
    m = max(len(b["caption_ids"]) for b in batch)
    caps = torch.zeros((len(batch), m), dtype=torch.long)
    for i,b in enumerate(batch):
        caps[i,:len(b["caption_ids"])] = b["caption_ids"]
    return {"images": images, "captions": caps}
