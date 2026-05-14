"""VQA v2 data loading: MSCOCO images, questions, and crowd answers.

Role in *Image captioning improved visual question answering*
-------------------------------------------------------------
This package prepares **(image, question, answers)** tuples consumed by ``VQAModel``. It mirrors
the VQA v2 annotation protocol used when evaluating caption-augmented systems in the thesis.

Dataset layout (config-driven)
------------------------------
YAML supplies **separate** MSCOCO train and validation files, for example::

    train_questions_json, train_annotations_json, train_images_dir
    val_questions_json, val_annotations_json, val_images_dir

plus ``train_image_filename_template`` / ``val_image_filename_template`` (``str.format`` with
``image_id``). No random 80/20 split over a single validation file.

Paper references
----------------
- **Vocabulary construction** — ``build_vocabs`` fits on official **training** questions/answers only.
- **Evaluation targets** — ``answers`` (ten annotators) + ``mode_answer`` for supervised token +
  ``vqa_acc`` soft scoring align with standard VQA v2 practice cited in thesis benchmarks.

Examples
--------
Building loaders::

    from datasets.vqa_dataset import all_qids, build_vocabs, VQADataset, collate

    tr_q = "/data/v2_OpenEnded_mscoco_train2014_questions.json"
    tr_a = "/data/v2_mscoco_train2014_annotations.json"
    qv, av = build_vocabs(tr_q, tr_a, min_freq=4)
    train_ds = VQADataset(
        tr_q,
        tr_a,
        "/data/train2014",
        "COCO_train2014_{image_id:012d}.jpg",
        qv,
        av,
        max_q=14,
        max_a=6,
        qids=all_qids(tr_q),
    )

Single-sample keys::

    sample = train_ds[0]
    # sample["image"]     — FloatTensor (3,448,448)
    # sample["q"]         — LongTensor question ids [BOS, ..., EOS]
    # sample["a"]        — LongTensor answer ids (mode answer tokenized)
    # sample["answers"] — raw strings from JSON for accuracy metric
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

TOKEN_RE = re.compile(r"[a-z0-9']+")


def tok(text: str) -> List[str]:
    """Tokenize text for vocab counting / encoding (lowercase alphanumerics + apostrophe).

    Examples:
        >>> tok("How many dogs?")
        ['how', 'many', 'dogs']
        >>> tok("it's red")
        ["it's", 'red']
    """
    return TOKEN_RE.findall(text.lower())


class Vocab:
    """Closed vocabulary with PAD/BOS/EOS/UNK plus frequency-filtered words.

    Answer vocab typically uses ``min_freq=1`` so rare modes remain reachable; question vocab may
    filter rare tokens via ``min_freq`` from YAML.

    Examples:
        >>> v = Vocab(["cat", "cat", "dog"], min_freq=2)
        >>> v.itos[:6]
        ['<pad>', '<bos>', '<eos>', '<unk>', 'cat']
        >>> v.encode(["cat", "zebra"])  # zebra unknown
        [4, 3]
    """

    PAD = "<pad>"
    BOS = "<bos>"
    EOS = "<eos>"
    UNK = "<unk>"

    def __init__(self, words: List[str], min_freq: int = 4) -> None:
        """Build token tables from a flat word list and frequency threshold.

        Args:
            words: All tokens contributing to counts (questions or answers).
            min_freq: Minimum occurrences required to keep a word out of UNK.
        """
        from collections import Counter

        c = Counter(words)
        self.itos = [self.PAD, self.BOS, self.EOS, self.UNK] + sorted([w for w, n in c.items() if n >= min_freq])
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, words: List[str]) -> List[int]:
        """Map whitespace-split tokens to indices with UNK fallback."""
        return [self.stoi.get(w, self.stoi[self.UNK]) for w in words]

    @property
    def pad_id(self) -> int:
        """Index of ``PAD``; labels padded with this index are ignored in cross-entropy."""
        return self.stoi[self.PAD]


def mode_answer(ans: List[str]) -> str:
    """Return the most common normalized answer among ten crowd votes.

    VQA provides ten answers per question; using the mode stabilizes the supervised token sequence.

    Examples:
        >>> mode_answer(["yes", "Yes", "yes", "no", "yes", "maybe", "yes", "Yes", "yes", "yes"])
        'yes'
    """
    return Counter([a.strip().lower() for a in ans]).most_common(1)[0][0]


def all_qids(questions_json: str) -> List[int]:
    """Collect every ``question_id`` from a VQA v2 questions JSON file.

    Args:
        questions_json: Path to ``v2_OpenEnded_mscoco_*_questions.json``.
    """
    with Path(questions_json).open("r", encoding="utf-8") as f:
        qs = json.load(f)["questions"]
    return [int(x["question_id"]) for x in qs]


def build_vocabs(questions_json: str, annotations_json: str, min_freq: int) -> Tuple[Vocab, Vocab]:
    """Fit question and answer vocabs using all (question, annotation) pairs in the **train** files.

    Args:
        questions_json: Training questions JSON path.
        annotations_json: Training annotations JSON path (parallel to questions).
        min_freq: Minimum frequency for **question** vocab inclusion.

    Returns:
        ``(question_vocab, answer_vocab)``.
    """
    with Path(questions_json).open("r", encoding="utf-8") as f:
        qs = json.load(f)["questions"]
    with Path(annotations_json).open("r", encoding="utf-8") as f:
        anns = json.load(f)["annotations"]
    qm = {int(x["question_id"]): x for x in qs}
    am = {int(x["question_id"]): x for x in anns}
    qids = sorted(set(qm.keys()) & set(am.keys()))
    qw, aw = [], []
    for qid in qids:
        qw.extend(tok(qm[qid]["question"]))
        aw.extend(tok(mode_answer([z["answer"] for z in am[qid]["answers"]])))
    return Vocab(qw, min_freq=min_freq), Vocab(aw, min_freq=1)


class VQADataset(Dataset):
    """PyTorch ``Dataset`` of VQA samples with ImageNet-normalized ``448×448`` tensors.

    Encoding convention:

    - Question tensor is ``[1] + encode(question tokens) + [2]`` (BOS index ``1``, EOS ``2``, PAD ``0``).
    - Answer tensor mirrors the same framing with ``max_a``.

    Paper tie-in: feeding ``q`` token ids into both ``VQAModel`` **and** the frozen captioner keeps
    the pathway aligned with question-conditioned captioning described in the thesis.

    Examples:
        >>> # ds = VQADataset(q_json, a_json, img_dir, tpl, qv, av, 14, 6, qids=all_qids(q_json))
    """

    def __init__(
        self,
        questions_json: str,
        annotations_json: str,
        images_dir: str,
        image_filename_template: str,
        qv: Vocab,
        av: Vocab,
        max_q: int = 14,
        max_a: int = 6,
        qids: Optional[List[int]] = None,
    ) -> None:
        """Load JSON annotations and build the in-memory sample list.

        Args:
            questions_json: Path to split-specific VQA questions file.
            annotations_json: Path to split-specific VQA annotations file.
            images_dir: Folder with COCO images for this split.
            image_filename_template: ``str.format`` with ``image_id`` (e.g. COCO filename).
            qv: Question vocabulary for encoding token ids.
            av: Answer vocabulary for the mode-answer supervision target.
            max_q: Max question tokens between BOS/EOS (caps raw tokens at ``max_q - 2``).
            max_a: Max answer tokens between BOS/EOS (caps at ``max_a - 2``).
            qids: Question ids to include; default is intersection of ids in both JSONs.
        """
        self.images_dir = Path(images_dir)
        self.image_filename_template = image_filename_template
        self.qv = qv
        self.av = av
        self.max_q = max_q
        self.max_a = max_a
        with Path(questions_json).open("r", encoding="utf-8") as f:
            qs = json.load(f)["questions"]
        with Path(annotations_json).open("r", encoding="utf-8") as f:
            anns = json.load(f)["annotations"]
        qm = {int(x["question_id"]): x for x in qs}
        am = {int(x["question_id"]): x for x in anns}
        use_qids = sorted(qids) if qids is not None else sorted(set(qm.keys()) & set(am.keys()))
        self.samples = []
        for qid in use_qids:
            if qid not in qm or qid not in am:
                continue
            q = qm[qid]
            a = am[qid]
            answers = [x["answer"] for x in a["answers"]]
            self.samples.append(
                {"qid": qid, "image_id": int(q["image_id"]), "question": q["question"], "answers": answers, "answer": mode_answer(answers)}
            )
        self.tf = transforms.Compose(
            [
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        """Number of (image, question, answers) samples in this split."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return one sample: resized/normalized image, padded token tensors, and raw answer list.

        Args:
            idx: Index into ``self.samples`` (0 .. ``len(self)-1``).

        Returns:
            Dict with keys ``image`` (CHW float), ``q`` and ``a`` (1D long tensors), and
            ``answers`` (list of ten annotator strings for ``vqa_acc``).
        """
        s = self.samples[idx]
        name = self.image_filename_template.format(image_id=s["image_id"])
        p = self.images_dir / name
        image = self.tf(Image.open(p).convert("RGB"))
        q = [1] + self.qv.encode(tok(s["question"])[: self.max_q - 2]) + [2]
        a = [1] + self.av.encode(tok(s["answer"])[: self.max_a - 2]) + [2]
        return {"image": image, "q": torch.tensor(q, dtype=torch.long), "a": torch.tensor(a, dtype=torch.long), "answers": s["answers"]}


def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pad batched question / answer token tensors with PAD ``0``.

    Args:
        batch: List of dicts from ``VQADataset.__getitem__``.

    Returns:
        Dict with keys ``images``, ``q``, ``a``, ``answers`` (lists of raw answer lists).

    Examples:
        >>> # dl = DataLoader(ds, batch_size=8, collate_fn=collate)
    """
    images = torch.stack([x["image"] for x in batch])
    qm = max(len(x["q"]) for x in batch)
    am = max(len(x["a"]) for x in batch)
    q = torch.zeros((len(batch), qm), dtype=torch.long)
    a = torch.zeros((len(batch), am), dtype=torch.long)
    for i, x in enumerate(batch):
        q[i, : len(x["q"])] = x["q"]
        a[i, : len(x["a"])] = x["a"]
    return {"images": images, "q": q, "a": a, "answers": [x["answers"] for x in batch]}
