"""Model package — Stage-1 QD captioner ha + registry.

Finglish — registry:
    ``model_class`` dar YAML esm class ro migire. ``build_captioner(cfg, vocab, q_vocab)``
    model misaze ta train / eval / benchmark / SimpleVQA ``load_captioner`` hame
    yek builder dashte bashan.

    - ``MMTransformerCopyCaptioner`` — pishfarz v2 (config haye ``configs/*.yaml``).
    - ``QuestionDependentImageCaptioner`` (v1) — ``DEFAULT_MODEL_CLASS``: checkpoint
      haye ghadimi ``model_class`` nadaran → v1 load mishan (backward compat).
"""

from typing import Any, Callable, Dict

from torch import nn

from .beam_utils import build_copy_map
from .captioner_mmtransformer_copy import MMTransformerCopyCaptioner
from .captioner_v1 import QuestionDependentImageCaptioner

DEFAULT_MODEL_CLASS = "QuestionDependentImageCaptioner"


def _build_v1(cfg: Dict[str, Any], vocab: Any, q_vocab: Any) -> nn.Module:
    hidden_dim = int(cfg.get("hidden_dim", cfg.get("lstm_hidden", 512)))
    return QuestionDependentImageCaptioner(
        vocab_size=len(vocab.itos),
        pad_id=vocab.pad_id,
        word_dim=int(cfg["word_dim"]),
        hidden_dim=hidden_dim,
        max_regions=int(cfg["max_regions"]),
        question_dim=int(cfg.get("question_dim", cfg["word_dim"])),
        embed_dim=int(cfg.get("embed_dim", hidden_dim)),
        region_dim=int(cfg.get("region_dim", 2048)),
        question_vocab_size=len(q_vocab.itos),
        question_pad_id=q_vocab.pad_id,
        dropout=float(cfg.get("dropout", 0.5)),
        fuse_mode=str(cfg.get("fuse_mode", "concat")),
    )


def _build_mmtransformer_copy(
    cfg: Dict[str, Any], vocab: Any, q_vocab: Any
) -> nn.Module:
    return MMTransformerCopyCaptioner(
        vocab_size=len(vocab.itos),
        pad_id=vocab.pad_id,
        question_vocab_size=len(q_vocab.itos),
        copy_ids=build_copy_map(vocab, q_vocab),
        question_pad_id=q_vocab.pad_id,
        d_model=int(cfg.get("hidden_dim", 512)),
        num_heads=int(cfg.get("num_heads", 8)),
        num_layers=int(cfg.get("num_layers", 3)),
        ff_dim=int(cfg.get("ff_dim", 1024)),
        max_regions=int(cfg["max_regions"]),
        region_dim=int(cfg.get("region_dim", 2048)),
        dropout=float(cfg.get("dropout", 0.2)),
    )


MODEL_REGISTRY: Dict[str, Callable[[Dict[str, Any], Any, Any], nn.Module]] = {
    DEFAULT_MODEL_CLASS: _build_v1,
    "MMTransformerCopyCaptioner": _build_mmtransformer_copy,
}


def build_captioner(cfg: Dict[str, Any], vocab: Any, q_vocab: Any) -> nn.Module:
    """Finglish: ``cfg['model_class']`` → builder; vocab/q_vocab = Vocab (itos/pad_id)."""
    name = str(cfg.get("model_class") or DEFAULT_MODEL_CLASS)
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_class {name!r}; known: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name](cfg, vocab, q_vocab)


__all__ = [
    "QuestionDependentImageCaptioner",
    "MMTransformerCopyCaptioner",
    "MODEL_REGISTRY",
    "build_captioner",
    "DEFAULT_MODEL_CLASS",
]
