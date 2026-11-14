"""Packed visual cache (root module, models + scripts import mikonan): hame image ha dar yek memmap (fp16) → RAM-e process ha share mishe.

Finglish — chera?
    Cache-e ``{image_id}.pt`` + ``_mem_cache`` dar CachedVisualEncoder baraye ~80k image (BUTD,
    37 x 2048 fp16 ≈ 150 KB) har process ~12 GB RAM migire; ba 2 process-e DDP roye Kaggle
    (~30 GB RAM) ja nemishe. Memmap az page cache-e OS khoonde mishe → yek nuskhe baraye hame.

    Layout-e ``<dir>``:
        meta.json   {"ids": [...], "tokens": T, "feat_dim": D}
        feats.f16   (N, T, D) float16
        boxes.f16   (N, T, 4) float16 (normalized xyxy)
        ntok.i16    (N,) tedad token-e vagheii (baghi zero = pad)
        done.u8     (N,) 1 = extract shode (resume)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

META = "meta.json"


def is_packed(cache_dir: str | Path) -> bool:
    return (Path(cache_dir) / META).exists()


def _arrays(d: Path, n: int, t: int, dim: int, mode: str):
    return (
        np.memmap(d / "feats.f16", np.float16, mode, shape=(n, t, dim)),
        np.memmap(d / "boxes.f16", np.float16, mode, shape=(n, t, 4)),
        np.memmap(d / "ntok.i16", np.int16, mode, shape=(n,)),
        np.memmap(d / "done.u8", np.uint8, mode, shape=(n,)),
    )


def init_packed(cache_dir: str | Path, ids: List[int], tokens: int, feat_dim: int) -> None:
    """Sakht-e file ha (zero, sparse). Age meta hast bayad yeksan bashe → resume."""
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    meta = {"ids": [int(i) for i in ids], "tokens": int(tokens), "feat_dim": int(feat_dim)}
    if (d / META).exists():
        old = json.loads((d / META).read_text())
        if old != meta:
            raise RuntimeError(f"{d}: existing packed cache has different ids/tokens/feat_dim")
        return
    for a in _arrays(d, len(ids), tokens, feat_dim, "w+"):
        a.flush()
    (d / META).write_text(json.dumps(meta))


class PackedWriter:
    def __init__(self, cache_dir: str | Path) -> None:
        self.dir = Path(cache_dir)
        meta = json.loads((self.dir / META).read_text())
        self.index = {i: k for k, i in enumerate(meta["ids"])}
        self.t, self.dim = int(meta["tokens"]), int(meta["feat_dim"])
        self.feats, self.boxes, self.ntok, self.done = _arrays(
            self.dir, len(meta["ids"]), self.t, self.dim, "r+")
        self._pending: List[int] = []

    def todo(self, ids: List[int]) -> List[int]:
        return [i for i in ids if not self.done[self.index[i]]]

    def write(self, image_id: int, feats, boxes) -> None:
        k = self.index[int(image_id)]
        n = min(len(feats), self.t)
        assert feats.shape[1] == self.dim, f"feat dim {feats.shape[1]} != {self.dim}"
        self.feats[k, :n] = feats[:n]
        self.feats[k, n:] = 0
        self.boxes[k, :n] = boxes[:n]
        self.boxes[k, n:] = 0
        self.ntok[k] = n
        self._pending.append(k)
        if len(self._pending) >= 256:
            self.flush()

    def flush(self) -> None:
        # Finglish: aval data flush, bad done flag → crash vasat = image dobare extract mishe
        self.feats.flush(); self.boxes.flush(); self.ntok.flush()
        for k in self._pending:
            self.done[k] = 1
        self.done.flush()
        self._pending = []


class PackedReader:
    """Lazy (har process memmap-e khodesh ro baz mikone, ba fork/spawn safe)."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.dir = Path(cache_dir)
        meta = json.loads((self.dir / META).read_text())
        self.index: Dict[int, int] = {i: k for k, i in enumerate(meta["ids"])}
        self.t, self.dim = int(meta["tokens"]), int(meta["feat_dim"])
        self._n = len(meta["ids"])
        self._arr: Optional[Tuple] = None

    def get(self, image_id: int):
        """→ (feats (n, D) float32 np, boxes (n, 4) float32 np)."""
        if self._arr is None:
            self._arr = _arrays(self.dir, self._n, self.t, self.dim, "r")
        k = self.index.get(int(image_id))
        feats, boxes, ntok, done = self._arr
        if k is None or not done[k]:
            raise RuntimeError(f"Packed cache miss: image {image_id} in {self.dir}")
        n = int(ntok[k])
        return np.asarray(feats[k, :n], np.float32), np.asarray(boxes[k, :n], np.float32)


def missing_ids(cache_dir: str | Path, ids) -> List[int]:
    """Finglish: image_id haye ke dar packed cache nistan ya hanuz extract nashodan."""
    meta = json.loads((Path(cache_dir) / META).read_text())
    index = {i: k for k, i in enumerate(meta["ids"])}
    done = np.memmap(Path(cache_dir) / "done.u8", np.uint8, "r", shape=(len(meta["ids"]),))
    return sorted(int(i) for i in set(ids) if i not in index or not done[index[i]])
