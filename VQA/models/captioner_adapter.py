"""Pul baraye load kardan captioner az checkpoint be dakhele ``VQAModel``.

In module ye bridge ast ke model captioning ro az ``SimpleImageCaptioner`` (ya legacy
``ImageCaptionerV1``) dynamic import mikone va ba VQA pipeline connect mikone.

Chera joda load mikonim?
------------------------
Captioner roye **caption vocabulary** (kalamat MSCOCO) train shode.
VQA amma **question vocabulary** (kalamat soal haye VQA) dare.
Ghablan ``word_emb`` baraye har do estefade mishod → size mismatch ya index eshtebah.

Hal:
    - ``word_emb`` + ``classifier`` → caption vocab (az checkpoint load)
    - ``q_emb`` (jadid) → question vocab (VQA ``q_ids``)

YAML config::

    captioner_project_root: ../SimpleImageCaptioner
    captioner_ckpt: ../SimpleImageCaptioner/outputs/default/best.pt
    captioner_class: SimpleImageCaptioner

Estefade dar ``training/train.py``::

    captioner = load_captioner(cfg, vocab_size=len(qv.itos), pad_id=qv.pad_id, device=device)
    model = VQAModel(len(qv.itos), len(av.itos), qv.pad_id, captioner, ...)

Nokte: bad az load, hame parameter haye captioner freeze hastan::

    assert all(not p.requires_grad for p in captioner.parameters())
"""

import importlib.util
import inspect
from pathlib import Path
from typing import Any, Dict

import torch


def _load_matching_state_dict(model: torch.nn.Module, state: Dict[str, torch.Tensor]) -> None:
    """Faghat weight hayi ke shape-shoon ba model match mikone ro az checkpoint load kon.

    ``strict=False`` tanha missing/unexpected key ro ignore mikone; age shape fargh dashte
    bashe hanooz error mide. In helper key haye mismatch (mesl ``q_emb`` ke toye checkpoint
    nist) ro skip mikone va baghiye (LSTM, attention, ``word_emb``, ...) ro load mikone.
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


def _caption_vocab_size_from_checkpoint(ckpt_path: Path) -> int:
    """Size vocabulary caption ro az ``best.pt`` / ``last.pt`` peyda kon.

    Aval az list ``vocab`` toye checkpoint mikhune; age nabood az shape
    ``word_emb.weight`` estefade mikone (radif = tedad kalamat caption).
    """
    state = torch.load(ckpt_path, map_location="cpu")
    vocab = state.get("vocab")
    if vocab is not None:
        return len(vocab)
    weight = state.get("model", state).get("word_emb.weight")
    if weight is not None:
        return int(weight.shape[0])
    raise ValueError(f"Cannot infer caption vocabulary size from {ckpt_path}")


def load_captioner(cfg: Dict[str, Any], vocab_size: int, pad_id: int, device: torch.device) -> torch.nn.Module:
    """Captioner ro besaz, weight haye caption ro load kon, freeze kon, be device befrest.

    Args:
        cfg: ``captioner_project_root``, ``captioner_ckpt``, ``captioner_class`` va
            hyperparameter haye lazem baraye ``SimpleImageCaptioner.__init__``.
        vocab_size: size **question vocabulary** VQA — baraye ``q_emb`` (na ``word_emb``).
        pad_id: index PAD soal (meslan 0) — hamoon convention ``VQADataset``.
        device: cuda ya cpu.

    Returns:
        Captioner dar halat ``eval()`` ke hame parameter hash ``requires_grad=False`` hast.

    Flow:
        1. class ro az ``captioner_v1.py`` import kon
        2. caption vocab size ro az checkpoint begir
        3. model ro ba ``vocab_size=caption_vocab`` + ``question_vocab_size=vocab_size`` besaz
        4. weight haye match-shode ro load kon (``q_emb`` random mimune chon toye ckpt nist)
        5. freeze + eval + to(device)

    Age checkpoint vojood nadashte bashe, caption layers ham ba question vocab size
    sakhte mishan (fallback — baraye smoke/debug).
    """
    p = Path(cfg["captioner_project_root"]).resolve() / "models" / "captioner_v1.py"
    spec = importlib.util.spec_from_file_location("cap_mod", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    cls = getattr(mod, cfg.get("captioner_class", "ImageCaptionerV1"))

    ck = Path(cfg["captioner_ckpt"])
    if ck.exists():
        caption_vocab_size = _caption_vocab_size_from_checkpoint(ck)
    else:
        caption_vocab_size = vocab_size

    init_kwargs: Dict[str, Any] = {
        "vocab_size": caption_vocab_size,
        "pad_id": pad_id,
        "word_dim": cfg["word_dim"],
        "hidden_dim": cfg["hidden_dim"],
        "max_regions": cfg["max_regions"],
        "question_dim": cfg["question_dim"],
    }
    params = inspect.signature(cls.__init__).parameters
    if "question_vocab_size" in params:
        init_kwargs["question_vocab_size"] = vocab_size
        init_kwargs["question_pad_id"] = pad_id

    m = cls(**init_kwargs)
    if ck.exists():
        st = torch.load(ck, map_location="cpu")
        _load_matching_state_dict(m, st.get("model", st))
        if "question_vocab_size" in params:
            print(
                f"Captioner loaded: caption_vocab={caption_vocab_size} "
                f"question_vocab={vocab_size}"
            )
    m.eval().to(device)
    for prm in m.parameters():
        prm.requires_grad = False
    return m
