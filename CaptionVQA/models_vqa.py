"""VQA baseline (Bottom-Up Top-Down, Anderson et al. 2018) and the same model + a caption branch.

Finglish:
    Baseline (use_caption=False):
        soal  → word emb (300, shared table) → GRU 512 → q
        tasvir → 37 BUTD region × 2048 → LN + Linear 512 → top-down attention ba q → v̂
        h = FC(q) ⊙ FC(v̂) → classifier (soft-score BCE)
    + Caption (use_caption=True), faghat in ezafe mishe:
        caption (BLIP, text) → hamoon word emb → **BiGRU** → state-e har kalame (SEQUENCE, na miangin:
        "not" dar state-e "running" mimune) → top-down attention ba q roye kalamat → ĉ
        g = σ(W [q; v̂; ĉ]) ;  h = FC(q) ⊙ (g ⊙ FC(v̂) + (1 − g) ⊙ FC(ĉ))
    Hame layer haye moshtarak (emb, GRU, region, attention, classifier) dar har do yeksan hastan.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from models import FC, Classifier, TopDownAttention

Batch = Dict[str, torch.Tensor]


class UpDnVQA(nn.Module):
    def __init__(self, vocab_size: int, n_answers: int, word_dim: int = 300, dim: int = 512,
                 region_dim: int = 2048, dropout: float = 0.2, use_caption: bool = False) -> None:
        super().__init__()
        self.use_caption = use_caption
        self.emb = nn.Embedding(vocab_size, word_dim, padding_idx=0)
        self.emb_drop = nn.Dropout(dropout)
        self.q_gru = nn.GRU(word_dim, dim, batch_first=True)
        self.v_proj = nn.Sequential(nn.LayerNorm(region_dim), nn.Linear(region_dim, dim), nn.ReLU(),
                                    nn.Dropout(dropout))
        self.v_att = TopDownAttention(dim, dropout)
        self.q_net, self.v_net = FC(dim, dim, dropout), FC(dim, dim, dropout)
        if use_caption:
            self.c_gru = nn.GRU(word_dim, dim // 2, batch_first=True, bidirectional=True)
            self.c_att = TopDownAttention(dim, dropout)
            self.c_net = FC(dim, dim, dropout)
            self.gate = nn.Linear(3 * dim, dim)
        self.classifier = Classifier(dim, n_answers, dropout)
        self.last_gate: Optional[torch.Tensor] = None
        self.last_cap_att: Optional[torch.Tensor] = None

    def _rnn(self, gru: nn.GRU, tokens: torch.Tensor):
        lengths = (tokens != 0).sum(1).clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(self.emb_drop(self.emb(tokens)), lengths.cpu(),
                                                   batch_first=True, enforce_sorted=False)
        out, h = gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=tokens.size(1))
        return out, h

    def forward(self, b: Batch) -> torch.Tensor:
        _, h = self._rnn(self.q_gru, b["q"])
        q = h[-1]
        regions = b["regions"].float()
        v_hat, _ = self.v_att(self.v_proj(regions), q, regions.abs().sum(-1) == 0)
        if not self.use_caption:
            return self.classifier(self.q_net(q) * self.v_net(v_hat))
        c_states, _ = self._rnn(self.c_gru, b["cap"])              # (B, Lc, dim): one state per caption word
        c_hat, att = self.c_att(c_states, q, b["cap"] == 0)
        g = torch.sigmoid(self.gate(torch.cat([q, v_hat, c_hat], dim=-1)))
        self.last_gate, self.last_cap_att = g.detach().mean(-1), att.detach()
        return self.classifier(self.q_net(q) * (g * self.v_net(v_hat) + (1 - g) * self.c_net(c_hat)))
