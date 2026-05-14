"""Abstract caption module shared with VQA via duck typing.

See ``VQA.models.base_captioner`` for full thesis-oriented commentary (*Image captioning improved
visual question answering*). Both modules intentionally expose identical method names so the VQA
project can ``importlib`` ``captioner_v1.py`` without a formal shared package.

Examples
--------
Subclass checklist::

    class MyCaptioner(BaseImageCaptioner):
        def generate_caption(...): ...
        def encode_caption(...): ...
        def get_caption_embedding(...): ...
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
from torch import nn


class BaseImageCaptioner(nn.Module, ABC):
    """Minimal caption surface required by ``VQAModel`` once dynamically imported.

    Subclasses implement greedy (or other) decoding, a sentence-level pooling of token ids, and
    the combined hook used during VQA forward passes. Keeps VQA code independent of the concrete
    caption architecture in ``captioner_v1.py``.
    """

    @abstractmethod
    def generate_caption(self, image: torch.Tensor, question_ids: Optional[torch.Tensor] = None, max_len: int = 20) -> torch.Tensor:
        """Produce caption token ids autoregressively (reference impl: greedy from BOS).

        Args:
            image: Batch ``(N, 3, H, W)``.
            question_ids: Optional padded question tokens for question-conditioned decoding.
            max_len: Maximum generated length including the start token.

        Returns:
            Long tensor ``(N, T)`` of token indices.
        """
        ...

    @abstractmethod
    def encode_caption(self, caption_ids: torch.Tensor) -> torch.Tensor:
        """Map token-id sequences to a fixed-size sentence embedding per image.

        Args:
            caption_ids: ``(N, T)`` caption tokens (e.g. from ``generate_caption``).

        Returns:
            Float tensor ``(N, D)`` fused later in ``VQAModel`` with visual attention.
        """
        ...

    @abstractmethod
    def get_caption_embedding(self, image: torch.Tensor, question_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run generation then pooling; primary entry point for frozen caption signal in VQA.

        Args:
            image: Batch ``(N, 3, H, W)``.
            question_ids: Same semantics as ``generate_caption``.

        Returns:
            ``(sentence_embedding, caption_token_ids)``.
        """
        ...
