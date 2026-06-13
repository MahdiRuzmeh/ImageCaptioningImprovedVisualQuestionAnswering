"""Dynamic import bridge from captioner checkpoint into ``VQAModel``.

Supports ``SimpleImageCaptioner`` (default) or legacy ``ImageCaptionerV1`` via YAML::

    captioner_project_root: ../SimpleImageCaptioner
    captioner_ckpt: ../SimpleImageCaptioner/outputs/default/best.pt
    captioner_class: SimpleImageCaptioner

Examples
--------
Inside ``training/train.py``::

    captioner = load_captioner(cfg, vocab_size=len(qv.itos), pad_id=qv.pad_id, device=device)
    model = VQAModel(len(qv.itos), len(av.itos), qv.pad_id, captioner, ...)

Frozen-parameter invariant::

    assert all(not p.requires_grad for p in captioner.parameters())
"""

import importlib.util
from pathlib import Path
from typing import Any, Dict

import torch


def load_captioner(cfg: Dict[str, Any], vocab_size: int, pad_id: int, device: torch.device) -> torch.nn.Module:
    """Load caption network class from disk, optionally restore weights, move to device, freeze.

    Args:
        cfg: Must include ``captioner_project_root``, ``captioner_ckpt``, hyperparameters used by
            ``SimpleImageCaptioner.__init__`` (or ``ImageCaptionerV1``).
        vocab_size: Size of **question** vocabulary for embedding layer resizing.
        pad_id: Padding index consistent with ``VQADataset`` specials.
        device: Target accelerator.

    Returns:
        Eval-mode captioner with ``requires_grad=False`` everywhere.

    Examples:
        Missing checkpoint still constructs random captioner (visual backbone pretrained by torchvision)::

            # if Path(cfg["captioner_ckpt"]).exists() is False — weights skipped silently
    """
    p = Path(cfg["captioner_project_root"]).resolve() / "models" / "captioner_v1.py"
    spec = importlib.util.spec_from_file_location("cap_mod", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    cls = getattr(mod, cfg.get("captioner_class", "ImageCaptionerV1"))
    m = cls(
        vocab_size=vocab_size,
        pad_id=pad_id,
        word_dim=cfg["word_dim"],
        hidden_dim=cfg["hidden_dim"],
        max_regions=cfg["max_regions"],
        question_dim=cfg["question_dim"],
    )
    ck = Path(cfg["captioner_ckpt"])
    if ck.exists():
        st = torch.load(ck, map_location="cpu")
        m.load_state_dict(st.get("model", st), strict=False)
    m.eval().to(device)
    for prm in m.parameters():
        prm.requires_grad = False
    return m
