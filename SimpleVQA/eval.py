"""SimpleVQA — eval va inference (marhale 2 paper).

In file joda az ``train.py`` hast. Ba ``best.pt`` mitooni:
- ``image_id`` + ``question`` bedi → **javab** begiri
- ya ``question_id`` bedi → image/soal/GT az JSON load beshe
- roye val split **VQA v2 accuracy** (greedy) + chand sample pred vs GT bebini

Chera in file?
--------------
``train.py --eval`` faghat accuracy kol ro chap mikone.
``eval.py`` baraye **test dasti** hast: yek soal, yek aks, ya chand mesal random.

Input / Output
--------------
- **Input:** ``image_id`` + matn soal, ya ``question_id`` az VQA JSON
- **Output:** javab (greedy decode) + question-conditioned caption + ground truth (mode answer) agar dar dataset bashe

Pish-niaz
---------
- Checkpoint VQA: ``outputs/<run>/best.pt`` (bayad ``model``, ``q_vocab``, ``a_vocab`` dashte bashe)
- Config hamooni ke train kardi (path dataset, ``captioner_ckpt``, ...)
- Run az folder ``SimpleVQA/``
- Python env ba ``torch`` + ``torchvision`` (mesl ``src/.venv``)

Vocab ha az **checkpoint** load mishan — hamoon index hayi ke train shode (na rebuild az JSON).

CLI arguments
-------------
``--config``       path be YAML (default: ``configs/default.yaml``)
``--ckpt``         **required** — VQA ``best.pt``
``--image-id``     COCO image_id (ba ``--question`` estefade mishe)
``--question``     matn soal (ba ``--image-id``)
``--question-id``  VQA question_id — image + soal + GT az JSON load mishe
``--split``        ``train`` ya ``val`` (default: ``val``) — baraye ``--image-id`` mode
``--samples``      ba val mode: tedad sample random (default 10)

Che mode ee ejra mishe?
-----------------------
1. **Ba question_id** — ``--question-id`` set shode (image/soal/GT az JSON)
2. **Ba image + soal** — har do ``--image-id`` va ``--question``
3. **Val metrics** — hich kodom nist → accuracy kol + samples

Chand mesal (local smoke)
-------------------------
::

    cd SimpleVQA

    # 1) image + soal → javab
    python eval.py --config configs/smoke.yaml --ckpt outputs/smoke/best.pt \\
        --image-id 262148 --question "Where is he looking?" --split val

    # 2) ba question_id (image + soal + GT az JSON)
    python eval.py --config configs/smoke.yaml --ckpt outputs/smoke/best.pt \\
        --question-id 262148000

    # 3) val accuracy + 10 sample
    python eval.py --config configs/smoke.yaml --ckpt outputs/smoke/best.pt \\
        --split val --samples 10

Chand mesal (Kaggle mini)
-------------------------
::

    python eval.py --config configs/kaggle_mini.yaml --ckpt outputs/kaggle_mini/best.pt \\
        --image-id 391895 --question "what color is the bus?" --split val

    python eval.py --config configs/kaggle_mini.yaml --ckpt outputs/kaggle_mini/best.pt \\
        --question-id 262148000

    python eval.py --config configs/kaggle_mini.yaml --ckpt outputs/kaggle_mini/best.pt \\
        --split val --samples 10
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from train import (
    PATH_KEYS,
    PROJECT_ROOT,
    VQADataset,
    Vocab,
    all_qids,
    build_vqa_model,
    cap_list,
    collate_batch,
    eval_epoch,
    load_config,
    mode_answer,
    resolve_path_fields,
    set_seed,
    tok,
)

def image_size_from_cfg(cfg: Dict[str, Any]) -> int:
    """Finglish: image_size ro az YAML migirim (default 448) ta ba train yeksan bashe."""
    return int(cfg.get("image_size", 448))


def image_transform(image_size: int) -> transforms.Compose:
    """Hamoon transform train: Resize(image_size) + ImageNet normalize."""
    size = int(image_size)
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def vocab_from_itos(itos: List[str]) -> Vocab:
    """``Vocab`` ro az list ``itos`` toye checkpoint dobare besaz.

    Chera lazeme?
        - VQA ``q_vocab`` va ``a_vocab`` toye ``best.pt`` save shodan
        - eval bayad hamoon index-ha ro estefade kone ke train shode

    Input:
        itos: list kalamat az checkpoint

    Output:
        Vocab object ba ``itos`` va ``stoi``
    """
    v = Vocab.__new__(Vocab)
    v.itos = list(itos)
    v.stoi = {w: i for i, w in enumerate(v.itos)}
    return v


def decode_ids(ids: List[int], vocab: Vocab) -> str:
    """Token id ha (soal ya javab) ro be matn tabdil kon.

    PAD/BOS/EOS ro skip mikonim. Baraye chap soal, javab, ya debug.
    """
    words = [
        vocab.itos[i]
        for i in ids
        if 0 < i < len(vocab.itos) and vocab.itos[i] not in (vocab.PAD, vocab.BOS, vocab.EOS)
    ]
    return " ".join(words).strip()


def load_vqa_checkpoint(
    cfg: Dict[str, Any], ckpt_path: Path, device: torch.device
) -> Tuple[torch.nn.Module, Vocab, Vocab]:
    """Model VQA + ``q_vocab``/``a_vocab`` ro az ``best.pt`` load kon.

    Flow:
        1. ``q_vocab``, ``a_vocab`` az checkpoint
        2. ``build_vqa_model`` (captioner + ResNet + GNN + LSTM, ...)
        3. ``load_state_dict`` kamel — shamel ``captioner.q_emb`` fine-tuned ham mishe

    Input:
        cfg: YAML config (path dataset, captioner_ckpt, hyperparams)
        ckpt_path: VQA ``best.pt``

    Output:
        (model, q_vocab, a_vocab) — model dar eval mode
    """
    state = torch.load(ckpt_path, map_location=device)
    if "q_vocab" not in state or "a_vocab" not in state:
        raise ValueError(f"Checkpoint must contain q_vocab and a_vocab: {ckpt_path}")

    q_vocab = vocab_from_itos(state["q_vocab"])
    a_vocab = vocab_from_itos(state["a_vocab"])
    model, _ = build_vqa_model(cfg, q_vocab, a_vocab, device)
    model.load_state_dict(state.get("model", state), strict=False)
    model.eval()
    return model, q_vocab, a_vocab


def encode_question_tensor(
    question: str, q_vocab: Vocab, max_len: int, device: torch.device
) -> torch.Tensor:
    """Matn soal ro encode kon: BOS + tokens + EOS.

    Hamoon convention ``VQADataset`` va train.

    Output:
        tensor (1, seq_len) roye device
    """
    ids = [1] + q_vocab.encode(tok(question)[: max_len - 2]) + [2]
    return torch.tensor([ids], dtype=torch.long, device=device)


def load_question_record(
    cfg: Dict[str, Any], question_id: int
) -> Tuple[int, str, List[str], str, str]:
    """``question_id`` ro dar val/train JSON peyda kon.

    Aval val ro check mikone, bad train — ta betooni har do split ro test koni.

    Input:
        question_id: mesl 262148000 (VQA v2 format)

    Output:
        image_id, matn soal, list 10 javab annotator, mode answer, split (``train``/``val``)

    Raises:
        KeyError: age question_id toye hich JSON nabashe
    """
    for split_name, q_json, a_json in (
        ("val", cfg["val_questions_json"], cfg["val_annotations_json"]),
        ("train", cfg["train_questions_json"], cfg["train_annotations_json"]),
    ):
        with Path(q_json).open("r", encoding="utf-8") as f:
            qs = {int(x["question_id"]): x for x in json.load(f)["questions"]}
        with Path(a_json).open("r", encoding="utf-8") as f:
            anns = {int(x["question_id"]): x for x in json.load(f)["annotations"]}
        if question_id not in qs or question_id not in anns:
            continue
        q = qs[question_id]
        ann = anns[question_id]
        answers = [x["answer"] for x in ann["answers"]]
        return (
            int(q["image_id"]),
            q["question"],
            answers,
            mode_answer(answers),
            split_name,
        )
    raise KeyError(f"question_id {question_id} not found in train/val JSON")


def load_image_tensor(
    image_id: int,
    images_dir: str,
    template: str,
    device: torch.device,
    image_size: int,
) -> torch.Tensor:
    """Yek COCO image ro load kon → tensor (1, 3, image_size, image_size).

    Input:
        image_id, images_dir, filename_template (mesl train config)
        image_size: az config (``image_size``) — hamoon abaad train

    Output:
        batch tensor roye device
    """
    path = Path(images_dir) / template.format(image_id=image_id)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return image_transform(image_size)(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def split_image_paths(cfg: Dict[str, Any], split: str) -> Tuple[str, str]:
    """Path folder image + filename template baraye train ya val.

    Output:
        (images_dir, filename_template)
    """
    if split == "train":
        return cfg["train_images_dir"], cfg["train_image_filename_template"]
    if split == "val":
        return cfg["val_images_dir"], cfg["val_image_filename_template"]
    raise ValueError(f"split must be train or val, got {split!r}")


def image_filename(image_id: int, template: str) -> str:
    """COCO image filename baraye chap dar eval (mesl ``COCO_val2014_000262148.jpg``)."""
    return template.format(image_id=image_id)


def load_caption_vocab(cfg: Dict[str, Any]) -> Optional[Vocab]:
    """Caption vocabulary ro az checkpoint captioner (stage 1) load kon."""
    if not bool(cfg.get("use_captioner", True)):
        return None
    ckpt_path = Path(cfg["captioner_ckpt"])
    if not ckpt_path.exists():
        return None
    state = torch.load(ckpt_path, map_location="cpu")
    vocab_itos = state.get("vocab")
    if vocab_itos is None:
        return None
    return vocab_from_itos(vocab_itos)


def max_caption_len_from_cfg(cfg: Dict[str, Any]) -> int:
    """``max_caption_len`` ro az config captioner checkpoint ya YAML begir."""
    if "max_caption_len" in cfg:
        return int(cfg["max_caption_len"])
    ckpt_path = Path(cfg.get("captioner_ckpt", ""))
    if ckpt_path.exists():
        cap_cfg = torch.load(ckpt_path, map_location="cpu").get("config", {})
        if "max_caption_len" in cap_cfg:
            return int(cap_cfg["max_caption_len"])
    return 20


def print_sample_report(
    *,
    image_id: int,
    image_file: str,
    question: str,
    pred_caption: str,
    pred_answer: str,
    answer: Optional[str] = None,
) -> None:
    """Chap yek sample VQA ba caption tolid shode."""
    print("---")
    print(f"img_id:       {image_id}")
    print(f"img_file:     {image_file}")
    print(f"question:     {question}")
    print(f"pred_caption: {pred_caption}")
    print(f"pred_answer:  {pred_answer}")
    if answer is not None:
        print(f"gt mode answer:  {answer}")
    print()


@torch.no_grad()
def predict_vqa_sample(
    model: torch.nn.Module,
    image: torch.Tensor,
    q: torch.Tensor,
    a_vocab: Vocab,
    cap_vocab: Optional[Vocab],
    cfg: Dict[str, Any],
) -> Tuple[str, str]:
    """Greedy answer + question-conditioned caption baraye yek (image, question)."""
    logits = model(
        image,
        q,
        a_ids=None,
        max_answer_len=int(cfg["max_answer_len"]),
    )
    pred_answer = decode_ids(logits.argmax(dim=-1)[0].tolist(), a_vocab)

    if (
        not bool(cfg.get("use_captioner", True))
        or cap_vocab is None
        or getattr(model, "captioner", None) is None
    ):
        pred_caption = "(captioner disabled)"
    else:
        cap_ids = model.captioner.generate_caption(
            image,
            q,
            max_caption_len_from_cfg(cfg),
        )
        pred_caption = decode_ids(cap_ids[0].tolist(), cap_vocab)

    return pred_answer, pred_caption


def run_single(
    model: torch.nn.Module,
    q_vocab: Vocab,
    a_vocab: Vocab,
    cap_vocab: Optional[Vocab],
    cfg: Dict[str, Any],
    device: torch.device,
    image_id: int,
    question: str,
    split: str,
    gt_answers: Optional[List[str]] = None,
    gt_mode: Optional[str] = None,
) -> None:
    """Yek sample VQA: image + soal → caption + javab (greedy decode) chap kon.

    Flow:
        1. image load
        2. soal encode
        3. captioner ``generate_caption(image, question)``
        4. ``model.forward`` ba ``a_ids=None`` → greedy answer decode
        5. chap pred + optional GT (age az ``--question-id`` omade)

    Input optional:
        gt_answers, gt_mode — baraye moghayese ba annotator ha
    """
    images_dir, template = split_image_paths(cfg, split)
    img_file = image_filename(image_id, template)
    image = load_image_tensor(
        image_id, images_dir, template, device, image_size_from_cfg(cfg)
    )
    q = encode_question_tensor(
        question, q_vocab, int(cfg["max_question_len"]), device
    )
    pred_answer, pred_caption = predict_vqa_sample(
        model, image, q, a_vocab, cap_vocab, cfg
    )
    print_sample_report(
        image_id=image_id,
        image_file=img_file,
        question=question,
        pred_caption=pred_caption,
        pred_answer=pred_answer,
        answer=gt_mode,
    )
    if gt_answers:
        print(f"gt_all: {gt_answers[:5]}{'...' if len(gt_answers) > 5 else ''}")
        print()


def build_val_loader(
    cfg: Dict[str, Any], q_vocab: Vocab, a_vocab: Vocab
) -> DataLoader:
    """DataLoader val besaz — vocab haye checkpoint (na rebuild az JSON).

    ``max_val_qids`` az config cap mikone (mesl smoke/kaggle_mini).
    Hamoon ``VQADataset`` + ``collate_batch`` train estefade mishe.
    """
    va_qids = cap_list(all_qids(cfg["val_questions_json"]), cfg.get("max_val_qids"))
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
        image_size=image_size_from_cfg(cfg),
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
    return DataLoader(val_ds, shuffle=False, **loader_kw)


def run_val_metrics(
    model: torch.nn.Module,
    q_vocab: Vocab,
    a_vocab: Vocab,
    cfg: Dict[str, Any],
    device: torch.device,
) -> None:
    """Greedy decode roye val → VQA v2 soft accuracy chap kon.

    Hamoon metric ``train.py`` eval: ``vqa_acc`` ba 10 javab annotator.
    Accuracy ro chap mikone (mesl ``0.1167`` baraye smoke).
    """
    val_loader = build_val_loader(cfg, q_vocab, a_vocab)
    acc = eval_epoch(
        model,
        val_loader,
        nn.CrossEntropyLoss(),
        a_vocab,
        cfg,
        device,
        greedy=True,
    )[1]
    print(f"Validation VQA accuracy (greedy, soft v2): {acc:.4f}")


def run_val_samples(
    model: torch.nn.Module,
    q_vocab: Vocab,
    a_vocab: Vocab,
    cap_vocab: Optional[Vocab],
    cfg: Dict[str, Any],
    device: torch.device,
    n: int,
) -> None:
    """N ta sample random az val — caption, soal, pred, gt ro chap kon.

    Seed az config (``seed: 42``) → har run hamoon sample ha (reproducible).
    Baraye didan model chikar mikone bedoon run kamel val.
    """
    template = cfg["val_image_filename_template"]
    va_qids = cap_list(all_qids(cfg["val_questions_json"]), cfg.get("max_val_qids"))
    ds = VQADataset(
        cfg["val_questions_json"],
        cfg["val_annotations_json"],
        cfg["val_images_dir"],
        cfg["val_image_filename_template"],
        q_vocab,
        a_vocab,
        int(cfg["max_question_len"]),
        int(cfg["max_answer_len"]),
        qids=va_qids,
        image_size=image_size_from_cfg(cfg),
    )
    rng = random.Random(int(cfg.get("seed", 42)))
    indices = rng.sample(range(len(ds)), min(n, len(ds)))

    print(f"\nSample predictions (n={len(indices)}):")
    for idx in indices:
        sample = ds.samples[idx]
        image_id = int(sample["image_id"])
        batch = collate_batch([ds[idx]])
        images = batch["images"].to(device)
        q = batch["q"].to(device)
        pred_answer, pred_caption = predict_vqa_sample(
            model, images, q, a_vocab, cap_vocab, cfg
        )
        gt = decode_ids(batch["a"][0].tolist(), a_vocab)
        print_sample_report(
            image_id=image_id,
            image_file=image_filename(image_id, template),
            question=sample["question"],
            pred_caption=pred_caption,
            pred_answer=pred_answer,
            answer=gt,
        )


def parse_args() -> argparse.Namespace:
    """Argument haye CLI ro parse kon.

    Bebin module docstring bala baraye jadval kamel argument-ha va mesal-ha.
    """
    p = argparse.ArgumentParser(description="SimpleVQA — eval / infer")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--ckpt", required=True, help="VQA checkpoint (.pt)")
    p.add_argument("--image-id", type=int, default=None)
    p.add_argument("--question", default=None, help="Question text (with --image-id)")
    p.add_argument(
        "--question-id",
        type=int,
        default=None,
        help="VQA question_id (loads image + question from JSON)",
    )
    p.add_argument("--split", choices=("train", "val"), default="val")
    p.add_argument(
        "--samples",
        type=int,
        default=0,
        help="With --split val and no single sample: show N examples",
    )
    return p.parse_args()


def main() -> None:
    """Entry point: config load → model load → yek sample ya val metrics.

    Priority:
        1. ``--question-id`` → load az JSON + GT
        2. ``--image-id`` + ``--question`` → manual input
        3. hich kodom → ``run_val_metrics`` + ``run_val_samples``
    """
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    cfg = load_config(str(config_path))
    resolve_path_fields(cfg, PATH_KEYS)
    set_seed(int(cfg.get("seed", 42)))

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.get("device") == "cuda" else "cpu"
    )
    ckpt_path = Path(args.ckpt).expanduser().resolve()
    model, q_vocab, a_vocab = load_vqa_checkpoint(cfg, ckpt_path, device)
    cap_vocab = load_caption_vocab(cfg)
    if bool(cfg.get("use_captioner", True)) and cap_vocab is None:
        print(
            "Warning: use_captioner=true but caption vocab not found in "
            f"{cfg.get('captioner_ckpt')} — pred_caption will be unavailable."
        )
    print(f"config={config_path}  ckpt={ckpt_path}  device={device}")

    if args.question_id is not None:
        image_id, question, answers, mode, split = load_question_record(
            cfg, args.question_id
        )
        run_single(
            model,
            q_vocab,
            a_vocab,
            cap_vocab,
            cfg,
            device,
            image_id,
            question,
            split,
            gt_answers=answers,
            gt_mode=mode,
        )
    elif args.image_id is not None and args.question:
        run_single(
            model,
            q_vocab,
            a_vocab,
            cap_vocab,
            cfg,
            device,
            args.image_id,
            args.question,
            args.split,
        )
    elif args.image_id is not None or args.question:
        raise SystemExit("Provide both --image-id and --question, or use --question-id.")
    else:
        run_val_metrics(model, q_vocab, a_vocab, cfg, device)
        n = args.samples if args.samples > 0 else 10
        run_val_samples(model, q_vocab, a_vocab, cap_vocab, cfg, device, n)


if __name__ == "__main__":
    main()
