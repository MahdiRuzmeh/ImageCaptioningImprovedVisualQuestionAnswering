"""Abstract interface between frozen caption weights and ``VQAModel``.

Paper / thesis linkage (*Image captioning improved visual question answering*)
------------------------------------------------------------------------------
Decouple concrete caption architectures from VQA code: ``VQAModel`` depends only on this minimal
API so alternate caption networks remain plug-compatible—mirror **modular diagram** blocks that
show caption submodule feeding fused multimodal representation.

Contract summary
----------------
Implementors live under ``ImageCaptioner/models/`` and are loaded at runtime via
``captioner_adapter.load_captioner``. Methods marked abstract correspond to caption generation +
sentence pooling used before ROI fusion.

Examples
--------
Concrete subclass::

    from models.captioner_v1 import ImageCaptionerV1
    captioner: BaseImageCaptioner = ImageCaptionerV1(...)

Inside ``VQAModel.forward``::

    # Frozen submodule evaluated without grads:
    with torch.no_grad():
        v_cap, generated_ids = self.captioner.get_caption_embedding(images, q_ids)
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
from torch import nn


class BaseImageCaptioner(nn.Module, ABC):
    """Abstract caption module consumed by ``VQAModel``.

    Methods
    -------
    generate_caption
        Autoregressive caption tokens (greedy in reference implementation).
    encode_caption
        Sentence-level embedding from token ids (mean pool over embeddings).
    get_caption_embedding
        Convenience wrapper chaining generation + pooling—primary hook referenced in thesis fusion.

    Paper reference
    ---------------
    Align abstract methods with caption pathway outputs described before **fusion with visual
    attention** (multiply/add in ``VQAModel``).
    """

    @abstractmethod
    def generate_caption(self, image: torch.Tensor, question_ids: Optional[torch.Tensor] = None, max_len: int = 20) -> torch.Tensor:
        ...

    @abstractmethod
    def encode_caption(self, caption_ids: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def get_caption_embedding(self, image: torch.Tensor, question_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        ...
