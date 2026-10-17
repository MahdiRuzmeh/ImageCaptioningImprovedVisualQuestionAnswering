"""Interface baraye captioner ke VQA module mitune load kone.

Paper §3.3–§3.4: caption module bayad betune caption generate kone va
be vector (v_cap) tabdil kone ta ba visual attention dar VQA fuse beshe.

QD Stage 1 (QuestionDependentImageCaptioner):
    - word_emb baraye caption, q_emb + q_gru baraye soal
    - soal lazeme; get_caption_embedding(..., differentiable=True) baraye
      train VQA ta grad be q_emb beresad
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
        """Caption generation az BOS — soal lazeme dar QD model."""

    @abstractmethod
    def encode_caption(self, caption_ids: torch.Tensor) -> torch.Tensor:
        """Mean-pool token embedding-ha → v_cap (paper §3.4)."""

    @abstractmethod
    def get_caption_embedding(
        self,
        image: torch.Tensor,
        question_ids: Optional[torch.Tensor] = None,
        differentiable: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate caption + pool → (v_cap, token_ids) baraye VQA fusion.

        Args:
            differentiable: age True (train VQA), v_cap bayad grad dashte bashe
                ta q_emb az answer loss update beshe. Implementation dar
                QuestionDependentImageCaptioner az LSTM hidden pool estefade
                mikone chon argmax gradient ro cut mikone.
        """
