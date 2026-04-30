import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

TOKEN_RE = re.compile(r"[a-z0-9']+")


def tok(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def split_qids(qids: List[int], seed: int = 42) -> Tuple[List[int], List[int]]:
    ids = sorted(qids)
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


def mode_answer(ans: List[str]) -> str:
    return Counter([a.strip().lower() for a in ans]).most_common(1)[0][0]


def all_qids(dataset_root: str) -> List[int]:
    with Path(dataset_root, "v2_OpenEnded_mscoco_val2014_questions.json").open("r", encoding="utf-8") as f:
        qs = json.load(f)["questions"]
    return [int(x["question_id"]) for x in qs]


def build_vocabs(dataset_root: str, train_qids: List[int], min_freq: int) -> Tuple[Vocab, Vocab]:
    root = Path(dataset_root)
    with (root / "v2_OpenEnded_mscoco_val2014_questions.json").open("r", encoding="utf-8") as f:
        qs = json.load(f)["questions"]
    with (root / "v2_mscoco_val2014_annotations.json").open("r", encoding="utf-8") as f:
        anns = json.load(f)["annotations"]
    qm = {int(x["question_id"]): x for x in qs}
    am = {int(x["question_id"]): x for x in anns}
    qw, aw = [], []
    for qid in train_qids:
        qw.extend(tok(qm[qid]["question"]))
        aw.extend(tok(mode_answer([z["answer"] for z in am[qid]["answers"]])))
    return Vocab(qw, min_freq=min_freq), Vocab(aw, min_freq=1)


class VQADataset(Dataset):
    def __init__(self, dataset_root: str, qids: List[int], qv: Vocab, av: Vocab, max_q: int = 14, max_a: int = 6) -> None:
        self.root = Path(dataset_root)
        self.qv = qv
        self.av = av
        self.max_q = max_q
        self.max_a = max_a
        with (self.root / "v2_OpenEnded_mscoco_val2014_questions.json").open("r", encoding="utf-8") as f:
            qs = json.load(f)["questions"]
        with (self.root / "v2_mscoco_val2014_annotations.json").open("r", encoding="utf-8") as f:
            anns = json.load(f)["annotations"]
        qm = {int(x["question_id"]): x for x in qs}
        am = {int(x["question_id"]): x for x in anns}
        self.samples = []
        for qid in qids:
            q = qm[qid]
            a = am[qid]
            answers = [x["answer"] for x in a["answers"]]
            self.samples.append({"qid": qid, "image_id": int(q["image_id"]), "question": q["question"], "answers": answers, "answer": mode_answer(answers)})
        self.tf = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        p = self.root / "val2014" / f"COCO_val2014_{s['image_id']:012d}.jpg"
        image = self.tf(Image.open(p).convert("RGB"))
        q = [1] + self.qv.encode(tok(s["question"])[: self.max_q - 2]) + [2]
        a = [1] + self.av.encode(tok(s["answer"])[: self.max_a - 2]) + [2]
        return {"image": image, "q": torch.tensor(q, dtype=torch.long), "a": torch.tensor(a, dtype=torch.long), "answers": s["answers"]}


def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([x["image"] for x in batch])
    qm = max(len(x["q"]) for x in batch)
    am = max(len(x["a"]) for x in batch)
    q = torch.zeros((len(batch), qm), dtype=torch.long)
    a = torch.zeros((len(batch), am), dtype=torch.long)
    for i, x in enumerate(batch):
        q[i,:len(x["q"])] = x["q"]
        a[i,:len(x["a"])] = x["a"]
    return {"images": images, "q": q, "a": a, "answers": [x["answers"] for x in batch]}
