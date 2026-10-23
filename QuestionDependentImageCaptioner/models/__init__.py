"""Model package — VQA `load_captioner` import mikone `models/captioner_v1.py`.

Finglish — registry:
    ``model_class`` dar YAML esm class ro migire (default v1). ``build_captioner``
    az cfg + vocab ha model misaze ta train/eval/benchmark hame yek builder dashte bashan.
"""

from typing import Any, Callable, Dict

from torch import nn

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


MODEL_REGISTRY: Dict[str, Callable[[Dict[str, Any], Any, Any], nn.Module]] = {
    DEFAULT_MODEL_CLASS: _build_v1,
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
    "MODEL_REGISTRY",
    "build_captioner",
    "DEFAULT_MODEL_CLASS",
]
