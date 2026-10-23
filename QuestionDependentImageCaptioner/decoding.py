"""Batched caption generation roye yek split — train (val CIDEr) + benchmark.

Finglish:
    Har model ``generate_caption(image, question_ids, max_len, beam_size=...,
    image_ids=..., region_cache_dir=...)`` dare (BaseImageCaptioner contract).
    Inja loader (shuffle=False) ro batch be batch decode mikonim va token ha ro
    ta avalin <eos> negah midarim.
"""

from __future__ import annotations

import time
from typing import List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

BOS_ID, EOS_ID, PAD_ID = 1, 2, 0


def ids_to_tokens(ids: Sequence[int], itos: Sequence[str]) -> List[str]:
    """Finglish: id → word; BOS/PAD skip; avalin EOS → stop (greedy ba'd az EOS ashghale)."""
    out: List[str] = []
    for i in ids:
        i = int(i)
        if i == EOS_ID:
            break
        if i in (BOS_ID, PAD_ID) or not (0 <= i < len(itos)):
            continue
        out.append(itos[i])
    return out


@torch.no_grad()
def generate_split_captions(
    model: torch.nn.Module,
    loader: DataLoader,
    itos: Sequence[str],
    device: torch.device,
    max_len: int,
    beam_size: int = 1,
    region_cache_dir: Optional[str] = None,
    length_alpha: float = 0.7,
    no_repeat_ngram: int = 3,
) -> Tuple[List[List[str]], float]:
    """Kol loader ro decode kon → (list token haye pred, seconds).

    Tartib khorooji = tartib dataset (loader bayad shuffle=False bashe).
    """
    was_training = model.training
    model.eval()
    hyps: List[List[str]] = []
    t0 = time.perf_counter()
    for batch in loader:
        images = batch["images"].to(device, non_blocking=device.type == "cuda")
        question_ids = batch["questions"].to(device)
        image_ids = batch["image_ids"].to(device)
        cap = model.generate_caption(
            images,
            question_ids,
            max_len,
            beam_size=beam_size,
            length_alpha=length_alpha,
            no_repeat_ngram=no_repeat_ngram,
            image_ids=image_ids,
            region_cache_dir=region_cache_dir,
            save_region_cache=True,
        )
        for row in cap.tolist():
            hyps.append(ids_to_tokens(row, itos))
    if device.type == "cuda":
        torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    if was_training:
        model.train()
    return hyps, secs
