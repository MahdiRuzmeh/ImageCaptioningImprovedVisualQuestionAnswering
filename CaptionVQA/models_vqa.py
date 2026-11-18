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


# ---------------------------------------------------------------------------
# Stronger model: MCAN-style encoder–decoder co-attention (Yu et al., CVPR 2019)
# ---------------------------------------------------------------------------
class _MHAtt(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.att = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)

    def forward(self, q, kv, kv_pad):
        return self.att(q, kv, kv, key_padding_mask=kv_pad, need_weights=False)[0]


class _FFN(nn.Module):
    def __init__(self, d: int, ff: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ff, d))

    def forward(self, x):
        return self.net(x)


class _SA(nn.Module):
    """Self-attention unit (post-LN, as in MCAN)."""

    def __init__(self, d, heads, ff, dropout):
        super().__init__()
        self.att, self.ffn = _MHAtt(d, heads, dropout), _FFN(d, ff, dropout)
        self.n1, self.n2, self.drop = nn.LayerNorm(d), nn.LayerNorm(d), nn.Dropout(dropout)

    def forward(self, x, pad):
        x = self.n1(x + self.drop(self.att(x, x, pad)))
        return self.n2(x + self.drop(self.ffn(x)))


class _SGA(nn.Module):
    """Self-attention + guided attention to the question (x attends to y)."""

    def __init__(self, d, heads, ff, dropout):
        super().__init__()
        self.sa, self.ga, self.ffn = _MHAtt(d, heads, dropout), _MHAtt(d, heads, dropout), _FFN(d, ff, dropout)
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, x_pad, y, y_pad):
        x = self.n1(x + self.drop(self.sa(x, x, x_pad)))
        x = self.n2(x + self.drop(self.ga(x, y, y_pad)))
        return self.n3(x + self.drop(self.ffn(x)))


class _AttFlat(nn.Module):
    """Attentional reduction: sequence → one vector (MCAN AttFlat, 1 glimpse)."""

    def __init__(self, d, dropout, out):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, 1))
        self.proj = nn.Linear(d, out)

    def forward(self, x, pad):
        a = self.mlp(x).squeeze(-1).float().masked_fill(pad, float("-inf"))
        a = torch.nan_to_num(torch.softmax(a, dim=-1), nan=0.0).to(x.dtype)
        return self.proj(torch.einsum("bl,bld->bd", a, x))


class MCANVQA(nn.Module):
    """MCAN-style VQA. Finglish:
        soal  → GloVe (300) → LSTM 512 → token ha → ENCODER: L× self-attention
        tasvir → 37 BUTD region × 2048 → Linear 512 (+ type "region")
        [+ caption → hamoon GloVe → LSTM-e joda 512 → token ha (+ type "caption")]  ← faghat dar model-e caption
        DECODER: L× (self-attention roye [region (; caption)] + guided attention be soal)
        AttFlat(soal) + AttFlat(region (; caption)) → LayerNorm → classifier.
    Baseline va caption model dagighan hamin model hastan; tafavot faghat token haye caption dar decoder-e.
    """

    def __init__(self, vocab_size: int, n_answers: int, word_dim: int = 300, d: int = 512, heads: int = 8,
                 ff: int = 2048, layers: int = 6, region_dim: int = 2048, dropout: float = 0.1,
                 use_caption: bool = False, emb_init: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.use_caption = use_caption
        self.emb = nn.Embedding(vocab_size, word_dim, padding_idx=0)
        if emb_init is not None:
            self.emb.weight.data.copy_(emb_init)
        self.q_lstm = nn.LSTM(word_dim, d, batch_first=True)
        self.v_proj = nn.Linear(region_dim, d)
        if use_caption:
            self.c_lstm = nn.LSTM(word_dim, d, batch_first=True)
            self.type_emb = nn.Embedding(2, d)  # 0 = region, 1 = caption word
        self.enc = nn.ModuleList(_SA(d, heads, ff, dropout) for _ in range(layers))
        self.dec = nn.ModuleList(_SGA(d, heads, ff, dropout) for _ in range(layers))
        self.flat_q, self.flat_x = _AttFlat(d, dropout, 2 * d), _AttFlat(d, dropout, 2 * d)
        self.norm = nn.LayerNorm(2 * d)
        self.classifier = nn.Linear(2 * d, n_answers)
        self.last_gate = None  # (UpDn interface) — MCAN has no gate

    def _lstm(self, lstm, tokens):
        out, _ = lstm(self.emb(tokens))
        return out

    def forward(self, b):
        q_tok = b["q"]
        y, y_pad = self._lstm(self.q_lstm, q_tok), q_tok == 0
        regions = b["regions"].float()
        x, x_pad = self.v_proj(regions), regions.abs().sum(-1) == 0
        if self.use_caption:
            c = self._lstm(self.c_lstm, b["cap"]) + self.type_emb.weight[1]
            x = torch.cat([x + self.type_emb.weight[0], c], dim=1)
            x_pad = torch.cat([x_pad, b["cap"] == 0], dim=1)
        for layer in self.enc:
            y = layer(y, y_pad)
        for layer in self.dec:
            x = layer(x, x_pad, y, y_pad)
        return self.classifier(self.norm(self.flat_q(y, y_pad) + self.flat_x(x, x_pad)))
