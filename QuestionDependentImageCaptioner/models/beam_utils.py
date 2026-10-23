"""Generic beam search baraye captioner haye jadid (state = dict tensor).

Finglish:
    v1 beam search be LSTM (h, c, fused) gره khorde. Inja ``step_fn(seqs, state)``
    → (logp (N*B, V), new_state) migirim ta ham RNN (faghat seqs[:, -1]) ham
    Transformer (kol prefix) estefade konan. Har tensor state ba dim0 = N*B
    ba beam reorder mishe. Raftar = v1: EOS-stop, length-norm, trigram blocking.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch

State = Dict[str, torch.Tensor]
StepFn = Callable[[torch.Tensor, State], Tuple[torch.Tensor, State]]


def block_repeat_ngram(logp: torch.Tensor, seqs: torch.Tensor, ngram: int) -> None:
    """Finglish — n-gram tekrari → -inf (vectorized roye tamam row ha).

    Baraye har row, har jaei ke (ngram-1) token akhar ghablan oomade, token ba'di
    oon occurrence ban mishe.
    """
    rows, length = seqs.shape
    if ngram <= 1 or length < ngram - 1:
        return
    k = ngram - 1
    prefix = seqs[:, -k:]  # (R, k)
    # windows: (R, length-k, k) — har window + token ba'dish
    if length - k <= 0:
        return
    windows = seqs.unfold(1, k, 1)[:, : length - k]  # (R, L-k, k)
    nxt = seqs[:, k:]  # (R, L-k)
    match = (windows == prefix.unsqueeze(1)).all(dim=-1)  # (R, L-k)
    if not bool(match.any()):
        return
    r_idx, pos = match.nonzero(as_tuple=True)
    logp[r_idx, nxt[r_idx, pos]] = float("-inf")


def expand_state(state: State, n: int, beam: int) -> State:
    """Finglish: har tensor (N, ...) → (N*beam, ...) ba repeat_interleave."""
    return {k: v.repeat_interleave(beam, dim=0) for k, v in state.items()}


@torch.no_grad()
def beam_search(
    step_fn: StepFn,
    state: State,
    n: int,
    device: torch.device,
    max_len: int,
    beam_size: int = 5,
    length_alpha: float = 0.7,
    no_repeat_ngram: int = 3,
    bos_id: int = 1,
    eos_id: int = 2,
    pad_id: int = 0,
) -> torch.Tensor:
    """Beam search → (N, T) token ids (BOS aval, PAD ba'd az EOS)."""
    beam = max(1, int(beam_size))
    state = expand_state(state, n, beam)
    beam_scores = torch.full((n, beam), float("-inf"), device=device)
    beam_scores[:, 0] = 0.0
    beam_scores = beam_scores.reshape(n * beam)
    seqs = torch.full((n * beam, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(n * beam, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        logp, state = step_fn(seqs, state)
        logp = logp.float().clone()
        vocab = logp.size(-1)
        if no_repeat_ngram > 0:
            block_repeat_ngram(logp, seqs, no_repeat_ngram)
        if bool(finished.any()):
            logp[finished] = float("-inf")
            logp[finished, pad_id] = 0.0
        total = (beam_scores.unsqueeze(-1) + logp).view(n, beam * vocab)
        top_scores, top_idx = total.topk(beam, dim=-1)
        beam_id = torch.div(top_idx, vocab, rounding_mode="floor")
        token_id = top_idx % vocab
        offset = (torch.arange(n, device=device) * beam).unsqueeze(-1)
        flat = (beam_id + offset).reshape(n * beam)
        state = {k: v[flat] for k, v in state.items()}
        seqs = seqs[flat]
        finished = finished[flat]
        beam_scores = top_scores.reshape(n * beam)
        tok = token_id.reshape(n * beam)
        seqs = torch.cat([seqs, tok.unsqueeze(1)], dim=1)
        finished = finished | (tok == eos_id)
        if bool(finished.all()):
            break

    lengths = (seqs != pad_id).sum(dim=1).clamp(min=1).float()
    norm = (beam_scores / lengths.pow(length_alpha)).view(n, beam)
    best = norm.argmax(dim=-1)
    seqs = seqs.view(n, beam, -1)
    return seqs[torch.arange(n, device=device), best]


def build_copy_map(vocab, q_vocab, unk_id: int = 3) -> torch.Tensor:
    """Finglish — q_vocab id → caption vocab id (copy). Special token ha → PAD (0)
    ta copy rooshon mass nade (PAD hamishe az copy mask mishe)."""
    ids = []
    for i, w in enumerate(q_vocab.itos):
        if i < 4:
            ids.append(0)
        else:
            ids.append(vocab.stoi.get(w, unk_id))
    return torch.tensor(ids, dtype=torch.long)


def mix_copy_distribution(
    vocab_logits: torch.Tensor,
    copy_attn: torch.Tensor,
    copy_ids: torch.Tensor,
    p_gen: torch.Tensor,
) -> torch.Tensor:
    """Finglish — pointer-generator: log(p_gen·softmax(logits) + (1-p_gen)·copy).

    vocab_logits (..., V); copy_attn (..., Lq) (sum=1 roye token haye copy-able);
    copy_ids (..., Lq) caption-vocab id; p_gen (..., 1). fp32 baraye AMP.
    Khorooji log-prob (..., V) — CrossEntropy roosh = NLL (log_softmax idempotent).
    """
    p_vocab = torch.softmax(vocab_logits.float(), dim=-1)
    p_gen = p_gen.float()
    copy = torch.zeros_like(p_vocab).scatter_add_(
        -1, copy_ids, copy_attn.float() * (1.0 - p_gen)
    )
    p = p_gen * p_vocab + copy
    return torch.log(p.clamp_min(1e-9))
