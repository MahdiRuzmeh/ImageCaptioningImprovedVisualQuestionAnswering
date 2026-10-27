"""Caption-aware VQA models A / B / C (classifier head, soft-score BCE).

Hame model ha yek voroudi-ye moshtarak migiran (``batch`` dict):
    q            (B, Lq)      soal token (VQA q_vocab, PAD=0)
    regions      (B, R, 1024) raw Faster R-CNN ROI (hamoon feature-e captioner)
    cap_tokens   (B, 3, Lc)   top-3 beam caption (caption vocab, PAD=0)
    cap_scores   (B, 3)       length-normalized beam log-prob
    cap_hidden   (B, Lc, d)   decoder state-e captioner baraye beam 1
    (C) cand_*                n-gram candidate haye caption (pointer head)

``use_caption=False`` → hamoon model bedoon caption (ablation).

A = gated top-down attention (image + caption)        — Anderson'18 / Wu & Mooney'19
B = single-stream Transformer, init az encoder-e captioner — VisualBERT / ViLT
C = A + pointer roye n-gram haye caption (M4C-style) + confidence gate
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

Batch = Dict[str, torch.Tensor]


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """mask=True → ignore. Row-e kamelan mask shode → weight 0 (na NaN)."""
    scores = scores.float().masked_fill(mask, float("-inf"))
    return torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)


class FC(nn.Sequential):
    def __init__(self, i: int, o: int, dropout: float = 0.0) -> None:
        super().__init__(nn.Linear(i, o), nn.ReLU(), nn.Dropout(dropout))


class TopDownAttention(nn.Module):
    """α_i = w·(FC(x_i) ⊙ FC(q)) ; pooled = Σ α_i x_i (Anderson et al. 2018)."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.x_fc, self.q_fc = FC(dim, dim), FC(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, q: torch.Tensor, mask: torch.Tensor):
        s = self.score(self.drop(self.x_fc(x) * self.q_fc(q).unsqueeze(1))).squeeze(-1)
        a = masked_softmax(s, mask)
        return torch.einsum("bk,bkd->bd", a.to(x.dtype), x), a


class QuestionEncoder(nn.Module):
    """Embedding (init az captioner q_emb) + GRU; khoruji = (q_vec, token states)."""

    def __init__(self, q_emb_init: torch.Tensor, dim: int, dropout: float) -> None:
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(q_emb_init.clone(), freeze=False, padding_idx=0)
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(q_emb_init.size(1), dim, batch_first=True)

    def forward(self, q: torch.Tensor):
        lengths = (q != 0).sum(1).clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            self.drop(self.emb(q)), lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        return h[-1]


class CaptionEncoder(nn.Module):
    """Caption token → [frozen captioner word_emb ; decoder state] → BiGRU states."""

    def __init__(self, word_emb: torch.Tensor, dim: int, dropout: float, n_beams: int = 1) -> None:
        super().__init__()
        d_cap = word_emb.size(1)
        self.register_buffer("word_emb", word_emb.clone() * math.sqrt(d_cap), persistent=False)
        self.in_norm = nn.LayerNorm(d_cap)
        self.hid_proj = nn.Sequential(nn.LayerNorm(d_cap), nn.Linear(d_cap, d_cap))
        self.beam_emb = nn.Embedding(n_beams, d_cap)
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(d_cap, dim // 2, batch_first=True, bidirectional=True)

    def forward(self, tokens: torch.Tensor, hidden: Optional[torch.Tensor], beam_idx: int = 0):
        """tokens (B, L); hidden (B, L, d) ya None → states (B, L, dim), mask (B, L)."""
        mask = tokens == 0
        x = self.in_norm(F.embedding(tokens.long(), self.word_emb)) + self.beam_emb.weight[beam_idx]
        if hidden is not None:
            x = x + self.hid_proj(hidden.float())
        x = self.drop(x) * (~mask).unsqueeze(-1)
        lengths = (~mask).sum(1).clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=tokens.size(1))
        return out, mask


class Classifier(nn.Sequential):
    def __init__(self, dim: int, n_answers: int, dropout: float) -> None:
        super().__init__(nn.Linear(dim, dim * 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim * 2, n_answers))


# ---------------------------------------------------------------------------
# A — gated top-down attention over image regions and caption words
# ---------------------------------------------------------------------------
class GatedUpDnVQA(nn.Module):
    def __init__(self, n_answers: int, q_emb_init: torch.Tensor, word_emb: torch.Tensor,
                 dim: int = 512, dropout: float = 0.2, use_caption: bool = True) -> None:
        super().__init__()
        self.use_caption = use_caption
        self.q_enc = QuestionEncoder(q_emb_init, dim, dropout)
        self.v_proj = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, dim), nn.ReLU(), nn.Dropout(dropout))
        self.v_att = TopDownAttention(dim, dropout)
        self.q_net, self.v_net = FC(dim, dim, dropout), FC(dim, dim, dropout)
        if use_caption:
            self.c_enc = CaptionEncoder(word_emb, dim, dropout)
            self.c_att = TopDownAttention(dim, dropout)
            self.c_net = FC(dim, dim, dropout)
            self.gate = nn.Linear(3 * dim, dim)
        self.classifier = Classifier(dim, n_answers, dropout)
        self.last_gate: Optional[torch.Tensor] = None

    def fuse(self, b: Batch):
        """Joint vector + caption token states (C pointer reuse mikone)."""
        q = self.q_enc(b["q"])
        regions = b["regions"].float()
        v = self.v_proj(regions)
        v_hat, _ = self.v_att(v, q, regions.abs().sum(-1) == 0)
        if not self.use_caption:
            return self.q_net(q) * self.v_net(v_hat), q, None, None
        c_states, c_mask = self.c_enc(b["cap_tokens"][:, 0], b["cap_hidden"])
        c_hat, _ = self.c_att(c_states, q, c_mask)
        g = torch.sigmoid(self.gate(torch.cat([q, v_hat, c_hat], dim=-1)))
        self.last_gate = g.detach().mean(-1)
        joint = self.q_net(q) * (g * self.v_net(v_hat) + (1 - g) * self.c_net(c_hat))
        return joint, q, c_states, c_mask

    def forward(self, b: Batch) -> torch.Tensor:
        return self.classifier(self.fuse(b)[0])


# ---------------------------------------------------------------------------
# B — single-stream Transformer over [CLS ; regions ; question ; caption]
# ---------------------------------------------------------------------------
class CaptionerInitTransformerVQA(nn.Module):
    """Encoder layer ha, region_proj, roi_to_region, pos/type embedding az captioner.

    Finglish: captioner hamin encoder ro baraye [region ; soal] train karde (grounding).
    Inja caption token ha (type jadid) + CLS ezafe mishan va kol encoder fine-tune mishe
    (lr-e kamtar baraye bakhsh-e pretrained, ``pretrained_param_names``).
    """

    def __init__(self, n_answers: int, q_emb_init: torch.Tensor, cap_state: Dict[str, torch.Tensor],
                 cap_cfg: dict, dropout: float = 0.2, use_caption: bool = True) -> None:
        super().__init__()
        d = int(cap_cfg["hidden_dim"])
        self.d, self.use_caption = d, use_caption
        self.roi_to_region = nn.Linear(1024, int(cap_cfg["region_dim"]))
        self.region_proj = nn.Sequential(nn.LayerNorm(int(cap_cfg["region_dim"])), nn.Linear(int(cap_cfg["region_dim"]), d))
        self.q_emb = nn.Embedding.from_pretrained(q_emb_init.clone(), freeze=False, padding_idx=0)
        self.q_pos = nn.Embedding(64, d)
        self.w_pos = nn.Embedding(64, d)
        self.word_emb = nn.Embedding(cap_state["word_emb.weight"].size(0), d, padding_idx=0)
        self.type_emb = nn.Embedding(2, d)        # 0 region, 1 question (pretrained)
        self.extra_type = nn.Embedding(2, d)      # 0 caption, 1 CLS (new)
        self.cls = nn.Parameter(torch.zeros(d))
        self.hid_proj = nn.Linear(d, d)
        nn.init.zeros_(self.hid_proj.weight)
        nn.init.zeros_(self.hid_proj.bias)
        layer = nn.TransformerEncoderLayer(d, int(cap_cfg["num_heads"]), int(cap_cfg["ff_dim"]),
                                           dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, int(cap_cfg["num_layers"]), norm=nn.LayerNorm(d),
                                             enable_nested_tensor=False)
        self.drop = nn.Dropout(dropout)
        self.classifier = Classifier(d, n_answers, dropout)
        self._load_captioner(cap_state)

    PRETRAINED = ("roi_to_region", "region_proj", "q_pos", "w_pos", "word_emb", "type_emb", "encoder")

    def _load_captioner(self, s: Dict[str, torch.Tensor]) -> None:
        mapping = {"region_encoder.roi_to_region.": "roi_to_region."}
        own = {}
        for k, v in s.items():
            for src, dst in mapping.items():
                if k.startswith(src):
                    own[dst + k[len(src):]] = v
            if k.split(".")[0] in ("region_proj", "q_pos", "w_pos", "word_emb", "type_emb", "encoder"):
                own[k] = v
        missing, unexpected = self.load_state_dict(own, strict=False)
        assert not unexpected, unexpected
        loaded = [k for k in self.state_dict() if k in own]
        assert any(k.startswith("encoder.layers.2") for k in loaded), "captioner encoder not loaded"

    def pretrained_param_names(self):
        return {n for n, _ in self.named_parameters() if n.split(".")[0] in self.PRETRAINED}

    def forward(self, b: Batch) -> torch.Tensor:
        regions = b["regions"].float()
        n = regions.size(0)
        r_mask = regions.abs().sum(-1) == 0
        r_tok = self.region_proj(self.roi_to_region(regions)) + self.type_emb.weight[0]
        q = b["q"]
        q_tok = self.q_emb(q) * math.sqrt(self.d) + self.q_pos(torch.arange(q.size(1), device=q.device)) + self.type_emb.weight[1]
        cls = (self.cls + self.extra_type.weight[1]).expand(n, 1, -1)
        toks = [cls, r_tok, q_tok]
        masks = [torch.zeros((n, 1), dtype=torch.bool, device=q.device), r_mask, q == 0]
        if self.use_caption:
            c = b["cap_tokens"][:, 0].long()
            c_tok = (self.word_emb(c) * math.sqrt(self.d) + self.w_pos(torch.arange(c.size(1), device=c.device))
                     + self.extra_type.weight[0] + self.hid_proj(b["cap_hidden"].float()))
            toks.append(c_tok)
            masks.append(c == 0)
        x = self.encoder(self.drop(torch.cat(toks, 1)), src_key_padding_mask=torch.cat(masks, 1))
        return self.classifier(x[:, 0])


# ---------------------------------------------------------------------------
# C — A + pointer over caption n-grams (open-vocabulary answers) + confidence gate
# ---------------------------------------------------------------------------
class CaptionPointerVQA(nn.Module):
    """Logits = [classifier (K) ; pointer roye M candidate] — M4C-style, BCE moshtarak.

    Candidate = n-gram (n≤3) az yeki az 3 beam caption. Representation = mean-e
    token state ha (BiGRU) dar span. Gate (confidence-e captioner: beam score ha +
    negation flag + tedad-e beam haei ke candidate toshun hast) yek bias roye
    hame pointer logit ha mizane → model yad migire key be caption etemad kone.
    """

    N_BEAMS = 3

    def __init__(self, n_answers: int, q_emb_init: torch.Tensor, word_emb: torch.Tensor,
                 neg_ids: torch.Tensor, dim: int = 512, dropout: float = 0.2, use_caption: bool = True) -> None:
        super().__init__()
        self.use_caption = use_caption
        self.base = GatedUpDnVQA(n_answers, q_emb_init, word_emb, dim, dropout, use_caption)
        if use_caption:
            self.beam_enc = CaptionEncoder(word_emb, dim, dropout, n_beams=self.N_BEAMS)
            self.register_buffer("neg_ids", neg_ids.long(), persistent=False)
            conf_dim = self.N_BEAMS * 2
            self.ptr_joint = nn.Linear(dim, dim)
            self.ptr_cand = nn.Sequential(nn.Linear(dim + 3, dim), nn.ReLU())
            self.gate = nn.Sequential(nn.Linear(dim + conf_dim, dim // 4), nn.ReLU(), nn.Linear(dim // 4, 1))
            self.neg_proj = nn.Linear(conf_dim, dim)
        self.last_gate: Optional[torch.Tensor] = None

    def forward(self, b: Batch) -> torch.Tensor:
        joint, q, _, _ = self.base.fuse(b)
        if not self.use_caption:
            return self.base.classifier(joint)
        toks = b["cap_tokens"].long()                                   # (B, 3, L)
        neg = torch.isin(toks, self.neg_ids).any(-1).float()            # (B, 3)
        conf = torch.cat([b["cap_scores"].float(), neg], dim=-1)        # (B, 6)
        joint = joint + self.neg_proj(conf)                             # yes/no: negation cue
        cls_logits = self.base.classifier(joint)

        states = []
        for k in range(self.N_BEAMS):
            hid = b["cap_hidden"] if k == 0 else None
            s, _ = self.beam_enc(toks[:, k], hid, beam_idx=k)
            states.append(s)
        states = torch.stack(states, 1)                                 # (B, 3, L, d)
        bsz, nb, L, d = states.shape
        flat = states.reshape(bsz, nb * L, d)
        # span matrix (B, M, 3L): 1/len roye token haye candidate
        pos = torch.arange(L, device=toks.device)
        start, length, beam = b["cand_start"], b["cand_len"], b["cand_beam"]
        in_span = (pos.view(1, 1, L) >= start.unsqueeze(-1)) & (pos.view(1, 1, L) < (start + length).unsqueeze(-1))
        span = torch.zeros((bsz, start.size(1), nb, L), device=toks.device)
        span.scatter_(2, beam.view(bsz, -1, 1, 1).expand(-1, -1, 1, L), in_span.float().unsqueeze(2))
        span = span.reshape(bsz, -1, nb * L) / length.clamp(min=1).unsqueeze(-1).float()
        cand = torch.bmm(span.to(flat.dtype), flat)                     # (B, M, d)
        cand = self.ptr_cand(torch.cat([cand, b["cand_feats"].to(cand.dtype)], -1))
        ptr = torch.einsum("bd,bmd->bm", self.ptr_joint(joint), cand) / math.sqrt(d)
        gate = self.gate(torch.cat([joint, conf.to(joint.dtype)], -1))  # (B, 1) logit bias
        self.last_gate = torch.sigmoid(gate.detach().squeeze(-1))
        ptr = (ptr + gate).float().masked_fill(~b["cand_mask"], -1e4)
        return torch.cat([cls_logits.float(), ptr], dim=-1)
