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
    """Minimal caption surface required by ``VQAModel`` once dynamically imported."""

    @abstractmethod
    def generate_caption(self, image: torch.Tensor, question_ids: Optional[torch.Tensor] = None, max_len: int = 20) -> torch.Tensor:
        ...

    @abstractmethod
    def encode_caption(self, caption_ids: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def get_caption_embedding(self, image: torch.Tensor, question_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        ...
