"""MMTransformerCopyCaptioner — multimodal Transformer (early fusion) + copy.

Architecture (benchmark arch B):
1. Faster R-CNN (freeze) → 32 region (RegionEncoder v1) → Linear → d.
2. Question tokens: q_emb + position → d. Har do + type embedding (region/question).
3. Joint encoder (pre-LN): region-ha va kalamat soal be ham attend mikonan →
   grounding dar har layer question-conditioned hast (na faghat yek attention).
4. Decoder: causal self-attention + cross-attention be memory (region + soal).
5. Copy head: attention az decoder state roye question part memory;
   p_gen*P_vocab + (1-p_gen)*copy (mesl arch A).

Finglish: train parallel (teacher forcing, bedoon scheduled sampling);
inference = beam search ba recompute kol prefix (caption kotahe, ≤20).
Output embedding ba word_emb tie shode (parameter kamtar baraye 3000 sample).
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
from torch import nn

from .base_captioner import BaseImageCaptioner
from .beam_utils import State, beam_search, mix_copy_distribution
from .captioner_v1 import RegionEncoder

SPECIAL_MAX_ID = 3  # PAD/BOS/EOS/UNK
MAX_POS = 64


class MMTransformerCopyCaptioner(BaseImageCaptioner):
    """Arch B: joint region+question Transformer encoder, Transformer decoder, copy."""

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        question_vocab_size: int,
        copy_ids: torch.Tensor,
        question_pad_id: int = 0,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
        ff_dim: int = 1024,
        max_regions: int = 32,
        region_dim: int = 2048,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.question_pad_id = int(question_pad_id)
        self.region_encoder = RegionEncoder(max_regions, region_dim)

        self.region_proj = nn.Sequential(
            nn.LayerNorm(region_dim), nn.Linear(region_dim, d_model)
        )
        self.q_emb = nn.Embedding(
            question_vocab_size, d_model, padding_idx=self.question_pad_id
        )
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        # Finglish: tied embedding → std = d^-0.5 (vagarna logits std ≈ sqrt(d));
        # input ha × sqrt(d) mishan ta scale ba position embedding hamahang bashe
        for emb, pad in ((self.q_emb, self.question_pad_id), (self.word_emb, pad_id)):
            nn.init.normal_(emb.weight, std=d_model**-0.5)
            with torch.no_grad():
                emb.weight[pad].zero_()
        self.q_pos = nn.Embedding(MAX_POS, d_model)
        self.w_pos = nn.Embedding(MAX_POS, d_model)
        self.type_emb = nn.Embedding(2, d_model)  # 0 = region, 1 = question
        self.emb_drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, ff_dim, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers, norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model, num_heads, ff_dim, dropout, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(
            dec_layer, num_layers, norm=nn.LayerNorm(d_model)
        )

        # Finglish: logits = h · word_emb^T (weight tying)
        self.out_bias = nn.Parameter(torch.zeros(vocab_size))
        self.copy_q = nn.Linear(d_model, d_model)
        self.copy_k = nn.Linear(d_model, d_model)
        self.p_gen = nn.Linear(3 * d_model, 1)
        self.register_buffer("copy_map", copy_ids.long(), persistent=False)
        # Identity baraye VQA loader compat (mesl v1)
        self.q_proj = nn.Identity()

    def train(self, mode: bool = True) -> "MMTransformerCopyCaptioner":
        super().train(mode)
        self.region_encoder.train(mode)
        return self

    # ------------------------------------------------------------------ encode
    def _encode(
        self,
        images: torch.Tensor,
        question_ids: torch.Tensor,
        image_ids: Optional[torch.Tensor],
        region_cache_dir: Optional[str],
        save_region_cache: bool,
    ) -> State:
        """Finglish: [region ; question] → joint encoder → memory + copy info."""
        regions = self.region_encoder(
            images,
            image_ids=image_ids,
            cache_dir=region_cache_dir,
            save_cache=save_region_cache,
        )
        r_mask = regions.abs().sum(dim=-1) == 0
        n, r = regions.shape[:2]
        lq = question_ids.size(1)
        r_tok = self.region_proj(regions) + self.type_emb.weight[0]
        q_pos = torch.arange(lq, device=question_ids.device)
        q_tok = (
            self.q_emb(question_ids) * math.sqrt(self.d_model)
            + self.q_pos(q_pos)
            + self.type_emb.weight[1]
        )
        q_pad = question_ids == self.question_pad_id
        mem_mask = torch.cat([r_mask, q_pad], dim=1)
        memory = self.encoder(
            self.emb_drop(torch.cat([r_tok, q_tok], dim=1)),
            src_key_padding_mask=mem_mask,
        )
        q_mem = memory[:, r:]
        return {
            "memory": memory,
            "mem_mask": mem_mask,
            "q_keys": self.copy_k(q_mem),
            "q_mem": q_mem,
            "copy_mask": question_ids <= SPECIAL_MAX_ID,
            "copy_ids": self.copy_map[question_ids],
        }

    # ------------------------------------------------------------------ decode
    def _decode(self, seqs: torch.Tensor, s: State) -> Tuple[torch.Tensor, torch.Tensor]:
        """Finglish: prefix (N,T) → (log-prob (N,T,V), hidden (N,T,d))."""
        t = seqs.size(1)
        pos = torch.arange(t, device=seqs.device)
        w = self.word_emb(seqs) * math.sqrt(self.d_model)
        x = self.emb_drop(w + self.w_pos(pos))
        causal = torch.triu(
            torch.ones(t, t, dtype=torch.bool, device=seqs.device), diagonal=1
        )
        h = self.decoder(
            x,
            s["memory"],
            tgt_mask=causal,
            memory_key_padding_mask=s["mem_mask"],
        )
        logits = h @ self.word_emb.weight.t() + self.out_bias
        scores = torch.einsum("btd,bld->btl", self.copy_q(h), s["q_keys"])
        scores = scores.float() / math.sqrt(self.d_model)
        scores = scores.masked_fill(s["copy_mask"].unsqueeze(1), float("-inf"))
        copy_attn = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
        ctx = torch.einsum("btl,bld->btd", copy_attn.to(h.dtype), s["q_mem"])
        p_gen = torch.sigmoid(self.p_gen(torch.cat([h, ctx, w], dim=-1)))
        copy_ids = s["copy_ids"].unsqueeze(1).expand(-1, t, -1)
        logp = mix_copy_distribution(logits, copy_attn, copy_ids, p_gen)
        return logp, h

    def forward_train(
        self,
        images: torch.Tensor,
        caption_ids: torch.Tensor,
        question_ids: Optional[torch.Tensor] = None,
        image_ids: Optional[torch.Tensor] = None,
        region_cache_dir: Optional[str] = None,
        save_region_cache: bool = True,
        scheduled_sampling_p: float = 0.0,
    ) -> torch.Tensor:
        """Predict caption_ids[:, 1:] — log-prob (N, T-1, V). scheduled_sampling ignore mishe."""
        if question_ids is None:
            raise ValueError("question_ids is required for MMTransformerCopyCaptioner")
        _ = scheduled_sampling_p
        s = self._encode(
            images, question_ids, image_ids, region_cache_dir, save_region_cache
        )
        logp, _ = self._decode(caption_ids[:, :-1], s)
        return logp

    # --------------------------------------------------------------- inference
    def _beam_step(self, seqs: torch.Tensor, s: State) -> Tuple[torch.Tensor, State]:
        logp, _ = self._decode(seqs, s)
        return logp[:, -1], s

    @torch.no_grad()
    def generate_caption(
        self,
        image: torch.Tensor,
        question_ids: Optional[torch.Tensor] = None,
        max_len: int = 20,
        beam_size: int = 5,
        length_alpha: float = 0.7,
        no_repeat_ngram: int = 3,
        image_ids: Optional[torch.Tensor] = None,
        region_cache_dir: Optional[str] = None,
        save_region_cache: bool = True,
    ) -> torch.Tensor:
        """Beam search (beam=1 → greedy) — soal lazeme."""
        if question_ids is None:
            raise ValueError("question_ids is required for MMTransformerCopyCaptioner")
        s = self._encode(
            image, question_ids, image_ids, region_cache_dir, save_region_cache
        )
        return beam_search(
            self._beam_step,
            s,
            image.size(0),
            image.device,
            max_len,
            beam_size=beam_size,
            length_alpha=length_alpha,
            no_repeat_ngram=no_repeat_ngram,
        )

    def encode_caption(self, caption_ids: torch.Tensor) -> torch.Tensor:
        """Mean-pool token embeddings → v_cap (N, d_model) — mesl v1."""
        return self.word_emb(caption_ids).mean(dim=1)

    def get_caption_embedding(
        self,
        image: torch.Tensor,
        question_ids: Optional[torch.Tensor] = None,
        differentiable: bool = False,
        image_ids: Optional[torch.Tensor] = None,
        region_cache_dir: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """VQA hook: eval → beam caption + mean word_emb; train → mean decoder state (EOS-aware)."""
        if question_ids is None:
            raise ValueError("question_ids is required for MMTransformerCopyCaptioner")
        if not differentiable:
            with torch.no_grad():
                cap = self.generate_caption(
                    image, question_ids, image_ids=image_ids,
                    region_cache_dir=region_cache_dir,
                )
            return self.encode_caption(cap), cap
        s = self._encode(image, question_ids, image_ids, region_cache_dir, True)
        seqs = torch.full((image.size(0), 1), 1, dtype=torch.long, device=image.device)
        for _ in range(19):
            with torch.no_grad():
                logp, _ = self._decode(seqs, s)
            seqs = torch.cat([seqs, logp[:, -1].argmax(-1, keepdim=True)], dim=1)
        # Finglish: yek pass ba grad roye kol caption greedy → hidden (N,19,d)
        _, h = self._decode(seqs[:, :-1], s)
        eos = seqs[:, 1:] == 2
        mask = ((eos.long().cumsum(1) - eos.long()) == 0).to(h.dtype).unsqueeze(-1)
        v_cap = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return v_cap, seqs.detach()
