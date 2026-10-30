"""Cached visual tokens + box geometry (visual-feature comparison: V0 / V1 / V2 / V3).

Finglish:
    ``scripts/extract_visual.py`` baraye har image yek ``{image_id}.pt`` mineviste:
      {"feats": (T, D) fp16, "boxes": (T, 4) fp16 normalized x1,y1,x2,y2 dar [0, 1]}
    - region backend (Faster R-CNN / BUTD): box = box-e region; token-e akhar = kol-e tasvir
    - grid backend (SigLIP / DINOv2): box = cell-e patch → hamoon box embedding = 2D position

    CachedVisualEncoder faghat az cache mikhoone (bedoon detector) va
    (N, max_tokens, D + 5) bar migardoone: [feats ; x1, y1, x2, y2, area].
    Pad token = hame sefr → captioner mask mikone. image_ablation (shuffle/mean/mask)
    roye hamin output kar mikone, pas box ham ba feature swap/mask mishe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

BOX_DIM = 5  # x1, y1, x2, y2, area


def box_geometry(boxes: torch.Tensor) -> torch.Tensor:
    """(T, 4) normalized xyxy → (T, 5) [x1, y1, x2, y2, area]."""
    area = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    return torch.cat([boxes, area[:, None]], dim=1)


class CachedVisualEncoder(nn.Module):
    """Cache-only visual encoder: (N,3,H,W) placeholder + image_ids → (N, T, D + 5)."""

    def __init__(self, max_tokens: int, feat_dim: int) -> None:
        super().__init__()
        self.max_tokens = int(max_tokens)
        self.feat_dim = int(feat_dim)
        # Finglish: RAM cache fp16 (CPU) — grid feature ha bozorgtaran (144 x 1152)
        self._mem_cache: Dict[str, torch.Tensor] = {}

    def _load(self, cache_dir: str, image_id: int) -> torch.Tensor:
        p = str(Path(cache_dir) / f"{int(image_id)}.pt")
        t = self._mem_cache.get(p)
        if t is None:
            if not Path(p).exists():
                raise RuntimeError(f"Visual cache miss: {p}; run scripts/extract_visual.py")
            d = torch.load(p, map_location="cpu")
            feats, boxes = d["feats"].float(), d["boxes"].float()
            if feats.ndim != 2 or feats.shape[1] != self.feat_dim:
                raise RuntimeError(f"{p}: feats {tuple(feats.shape)}, expected (T, {self.feat_dim})")
            feats, boxes = feats[: self.max_tokens], boxes[: self.max_tokens]
            t = torch.zeros(self.max_tokens, self.feat_dim + BOX_DIM)
            t[: len(feats)] = torch.cat([feats, box_geometry(boxes)], dim=1)
            t = t.half()
            self._mem_cache[p] = t
        return t

    def forward(
        self,
        images: torch.Tensor,
        image_ids: Optional[torch.Tensor] = None,
        cache_dir: Optional[str] = None,
        save_cache: bool = True,
    ) -> torch.Tensor:
        if not cache_dir or image_ids is None:
            raise RuntimeError("CachedVisualEncoder needs image_ids and a cache dir (images_from_cache: true)")
        out = torch.stack([self._load(cache_dir, int(i)) for i in image_ids.tolist()])
        return out.to(images.device, dtype=torch.float32, non_blocking=True)


class MultiCachedVisualEncoder(nn.Module):
    """Chand cache (e.g. BUTD + DINOv2) → (N, sum T_k, D_max + 5), source ha posht-e ham.

    Finglish: token haye source k dar baze-ye sabet [start_k, end_k) hastan; feature ta D_k
    por mishe (baghie sefr), box hamishe dar [D_max:]. Captioner baraye har source projection-e
    khodesh ro roye hamin baze mizane. ``cache_dir``-e voroodi ignore mishe (har source dir-e khodesh).
    """

    def __init__(self, sources: List[Tuple[str, int, int]]) -> None:
        """sources: (cache_dir, tokens, feat_dim) baraye har source."""
        super().__init__()
        root = Path(__file__).resolve().parents[1]
        self.dirs = [str(Path(d) if Path(d).is_absolute() else root / d) for d, _, _ in sources]
        self.encs = [CachedVisualEncoder(t, dim) for _, t, dim in sources]
        self.feat_dim = max(dim for _, _, dim in sources)
        self.spans: List[Tuple[int, int, int]] = []  # (start, end, feat_dim)
        s = 0
        for _, t, dim in sources:
            self.spans.append((s, s + t, dim))
            s += t
        self.max_tokens = s

    def forward(
        self,
        images: torch.Tensor,
        image_ids: Optional[torch.Tensor] = None,
        cache_dir: Optional[str] = None,
        save_cache: bool = True,
    ) -> torch.Tensor:
        if image_ids is None:
            raise RuntimeError("MultiCachedVisualEncoder needs image_ids")
        ids = image_ids.tolist()
        out = torch.zeros(len(ids), self.max_tokens, self.feat_dim + BOX_DIM, dtype=torch.float16)
        for enc, d, (s, e, dim) in zip(self.encs, self.dirs, self.spans):
            t = torch.stack([enc._load(d, int(i)) for i in ids])  # (N, T_k, dim + 5)
            out[:, s:e, :dim] = t[..., :dim]
            out[:, s:e, self.feat_dim:] = t[..., dim:]
        return out.to(images.device, dtype=torch.float32, non_blocking=True)
