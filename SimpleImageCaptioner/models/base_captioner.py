"""Interface baraye captioner ke VQA module mitune load kone.

Paper §3.3–§3.4: caption module bayad betune caption generate kone va
be vector (v_cap) tabdil kone ta ba visual attention dar VQA fuse beshe.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
from torch import nn


class BaseImageCaptioner(nn.Module, ABC):
    """Contract-e minimum ke VQAModel niyaz dare.

    VQA faghat in 3 method ro seda mizane:
    generate_caption, encode_caption, get_caption_embedding.
    """

    @abstractmethod
    def generate_caption(
        self,
        image: torch.Tensor,
        question_ids: Optional[torch.Tensor] = None,
        max_len: int = 20,
    ) -> torch.Tensor:
        """Greedy caption generation az BOS token."""

    @abstractmethod
    def encode_caption(self, caption_ids: torch.Tensor) -> torch.Tensor:
        """Mean-pool token embedding-ha → v_cap (paper §3.4)."""

    @abstractmethod
    def get_caption_embedding(
        self,
        image: torch.Tensor,
        question_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate caption + pool → (v_cap, token_ids) baraye VQA fusion."""
