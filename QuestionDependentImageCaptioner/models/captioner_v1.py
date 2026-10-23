"""QuestionDependentImageCaptioner — QD caption az (image, question).

Architecture (Stage 1):
1. Faster R-CNN (freeze) → ta 32 region × 1024; roi_to_region → 2048.
2. q_emb + GRU → qctx (soal feature).
3. Yek bar question attention: vazn har region az roye qctx → attended visual.
4. Feature fusion (concat): fused = Linear(concat(qctx, attended)).
5. LSTM decoder: h0/c0 az fused; har step attention faghat ba h_{t-1}
   (refocus region); input = concat(word, fused, z_t).

Finglish: RelationGNN inja nist. Soal lazeme (question_ids=None raise).
VQA hook (§3.4): get_caption_embedding → v_cap baraye Stage 2.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch import nn
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)

from .base_captioner import BaseImageCaptioner


class RegionEncoder(nn.Module):
    """Encoder: tasvir → ta 32 region vector (paper L=2048, K=32).

    Faster R-CNN freeze hast; faghat layer `roi_to_region` train mishe
    ta 1024-D ROI ro be 2048-D map kone (paper §3.1).
    """

    ROI_FEAT_DIM = 1024

    def __init__(self, max_regions: int = 32, region_dim: int = 2048) -> None:
        super().__init__()
        self.max_regions = max_regions
        self.region_dim = region_dim
        det = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        )
        self.detector = det

        for p in self.detector.parameters():
            p.requires_grad = False

        self.detector.eval()

        # Finglish — double-normalization fix:
        #   Dataset ghablan image ro ba ImageNet mean/std normalize mikone.
        #   GeneralizedRCNNTransform default dobare hamon normalize ro mizane → double!
        #   mean=0,std=1 mizarim ta detector faghat resize kone (na renormalize).
        self.detector.transform.image_mean = [0.0, 0.0, 0.0]
        self.detector.transform.image_std = [1.0, 1.0, 1.0]

        self.roi_to_region = nn.Linear(self.ROI_FEAT_DIM, region_dim)
        # Finglish: RAM cache (CPU tensor) ta har epoch file dobare az disk load nashe
        self._mem_cache: dict = {}

    def train(self, mode: bool = True) -> "RegionEncoder":
        """Detector hamishe eval mimune (BN freeze)."""
        super().train(mode)
        self.detector.eval()
        return self

    def _cache_path(self, cache_dir: str, image_id: int) -> str:
        return str(Path(cache_dir) / f"{int(image_id)}.pt")

    def _load_cached(
        self, cache_dir: str, image_id: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        """Finglish — cache raw ROI (1024D); roi_to_region bad az load train mishe."""
        try:
            p = self._cache_path(cache_dir, image_id)
            if p in self._mem_cache:
                return self._mem_cache[p].to(
                    device, non_blocking=(device.type == "cuda")
                )
            if not Path(p).exists():
                return None
            t = torch.load(p, map_location="cpu")
            if not isinstance(t, torch.Tensor):
                return None
            if (
                t.ndim != 2
                or t.shape[0] != self.max_regions
                or t.shape[1] != self.ROI_FEAT_DIM
            ):
                return None
            self._mem_cache[p] = t.float()
            return t.float().to(device, non_blocking=(device.type == "cuda"))
        except Exception:
            return None

    def _save_cached(
        self, cache_dir: str, image_id: int, regions: torch.Tensor
    ) -> None:
        """Finglish: region tensor ro roye CPU save kon."""
        try:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            p = self._cache_path(cache_dir, image_id)
            torch.save(regions.detach().float().to("cpu"), p)
        except Exception:
            pass

    def forward(
        self,
        images: torch.Tensor,
        image_ids: Optional[torch.Tensor] = None,
        cache_dir: Optional[str] = None,
        save_cache: bool = True,
    ) -> torch.Tensor:
        """(N,3,H,W) → (N, max_regions, region_dim).

        Finglish: backbone zir no_grad; roi_to_region gradient migire.
        Cache raw ROI 1024D; projection bad az load.
        """
        device = images.device
        n = images.size(0)

        if cache_dir and image_ids is not None and image_ids.numel() == n:
            cached: List[Optional[torch.Tensor]] = [
                self._load_cached(cache_dir, int(image_ids[i].item()), device)
                for i in range(n)
            ]
            if all(t is not None for t in cached):
                raw = torch.stack([t for t in cached if t is not None], dim=0)
                return self.roi_to_region(raw)

        if images.size(-1) < 32 or images.size(-2) < 32:
            # Finglish: placeholder image (images_from_cache) + cache miss → detector
            # roye 1x1 zero feature ashghal mide; behtare error bedim.
            raise RuntimeError(
                "Region cache miss for placeholder images; run scripts/cache_regions.py"
            )
        with torch.no_grad():
            img_list = list(images)
            transformed, _ = self.detector.transform(img_list, None)
            feats = self.detector.backbone(transformed.tensors)
            proposals, _ = self.detector.rpn(transformed, feats, None)
            roi = self.detector.roi_heads.box_roi_pool(
                feats, proposals, transformed.image_sizes
            )
            roi = self.detector.roi_heads.box_head(roi)
            counts = [len(p) for p in proposals]
            chunks = torch.split(roi, counts)
            batch_roi: List[torch.Tensor] = []
            for chunk in chunks:
                r = chunk[: self.max_regions]
                if r.size(0) < self.max_regions:
                    pad = torch.zeros(
                        (self.max_regions - r.size(0), r.size(1)), device=r.device
                    )
                    r = torch.cat([r, pad], dim=0)
                batch_roi.append(r)
            raw_roi = torch.stack(batch_roi, dim=0)

        if (
            cache_dir
            and image_ids is not None
            and image_ids.numel() == n
            and save_cache
        ):
            for i in range(n):
                self._save_cached(cache_dir, int(image_ids[i].item()), raw_roi[i])

        return self.roi_to_region(raw_roi)


class RegionAttention(nn.Module):
    """Soft attention rooye region-ha — query (N,D) × regions (N,R,2048) → context 512.

    Finglish: h_proj/v_proj → 512; score = dot; softmax; z 2048-D → ctx_proj 512.
    Age region pad (all-zero) bashe, score = -inf ta vazn nazdike.
    """

    def __init__(
        self,
        region_dim: int,
        query_dim: int,
        embed_dim: int = 512,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.h_proj = nn.Linear(query_dim, embed_dim)
        self.v_proj = nn.Linear(region_dim, embed_dim)
        self.ctx_proj = nn.Linear(region_dim, embed_dim)

    def forward(
        self, regions: torch.Tensor, query: torch.Tensor
    ) -> torch.Tensor:
        """regions (N,R,2048), query (N,query_dim) → context (N,512)."""
        h_att = self.h_proj(query)
        v_att = self.v_proj(regions)
        scores = (v_att * h_att.unsqueeze(1)).sum(dim=-1)
        # Finglish: pad region (norm≈0) → -inf ta softmax vazn nade
        pad_mask = regions.abs().sum(dim=-1) == 0
        scores = scores.masked_fill(pad_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        # age hame region pad bashe softmax NaN mide → zero context
        weights = torch.nan_to_num(weights, nan=0.0)
        z = torch.einsum("br,brd->bd", weights, regions)
        return self.ctx_proj(z)


class FeatureFusion(nn.Module):
    """Fuse qctx ba attended visual — alan faghat concat.

    Finglish: concat(qctx, attended) → Linear(2H → H).
    mode digar (add/mul) badeha ezafe mishe; alan reject.
    """

    def __init__(self, dim: int, mode: str = "concat") -> None:
        super().__init__()
        self.mode = str(mode).lower()
        if self.mode != "concat":
            raise ValueError(
                f"FeatureFusion only supports mode='concat', got {mode!r}"
            )
        self.proj = nn.Linear(dim * 2, dim)

    def forward(
        self, qctx: torch.Tensor, attended: torch.Tensor
    ) -> torch.Tensor:
        """qctx (N,D), attended (N,D) → fused (N,D)."""
        return self.proj(torch.cat([qctx, attended], dim=-1))


class QuestionDependentImageCaptioner(BaseImageCaptioner):
    """QD captioner: (image, question) → question-related caption.

    Flow:
        regions = FasterRCNN(image)
        qctx = GRU(q_emb(question))
        attended = question_attention(regions, qctx)   # yek bar
        fused = FeatureFusion.concat(qctx, attended)
        h0,c0 = tanh(W(fused))
        har step: z_t = step_attention(regions, h); LSTM([word; fused; z_t])

    Do vocabulary:
        - word_emb + classifier → caption tokens
        - q_emb + q_gru → question tokens (lazem)
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        word_dim: int = 512,
        hidden_dim: int = 512,
        max_regions: int = 32,
        question_dim: int = 1280,
        embed_dim: Optional[int] = None,
        region_dim: int = 2048,
        question_vocab_size: Optional[int] = None,
        question_pad_id: Optional[int] = None,
        dropout: float = 0.5,
        fuse_mode: str = "concat",
    ) -> None:
        """QD decoder + GRU question encoder + one-shot + per-step attention.

        Args:
            vocab_size: size vocabulary caption (word_emb, classifier).
            pad_id: PAD caption.
            question_vocab_size: lazeme — q_emb + q_gru.
            question_pad_id: PAD soal; default hamoon pad_id.
            fuse_mode: faghat ``concat`` alan.
        """
        super().__init__()
        if question_vocab_size is None:
            raise ValueError(
                "question_vocab_size is required for QuestionDependentImageCaptioner"
            )
        self.lstm_hidden = hidden_dim
        self.embed_dim = embed_dim if embed_dim is not None else hidden_dim
        self.word_dim = word_dim
        self.region_dim = region_dim
        self.question_pad_id = (
            pad_id if question_pad_id is None else int(question_pad_id)
        )

        self.region_encoder = RegionEncoder(max_regions, region_dim)

        # Finglish: question attention yek bar; step attention har kalame ba h.
        self.question_attention = RegionAttention(
            region_dim, hidden_dim, self.embed_dim
        )
        self.step_attention = RegionAttention(
            region_dim, hidden_dim, self.embed_dim
        )
        self.fusion = FeatureFusion(hidden_dim, mode=fuse_mode)

        self.word_emb = nn.Embedding(vocab_size, word_dim, padding_idx=pad_id)

        # Finglish: LSTM input = word + fused (QD) + z_t (per-step visual)
        self.lstm = nn.LSTMCell(
            word_dim + hidden_dim + self.embed_dim, hidden_dim
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, vocab_size)

        # Finglish: deep-output — logit = f(h, z_t) ta vision shortcut bemune
        self.ctx_to_logit = nn.Linear(self.embed_dim, hidden_dim)

        # Finglish: LSTM init az fused (soal + attended visual)
        self.init_h = nn.Linear(hidden_dim, hidden_dim)
        self.init_c = nn.Linear(hidden_dim, hidden_dim)

        self.q_emb = nn.Embedding(
            question_vocab_size, word_dim, padding_idx=self.question_pad_id
        )
        self.q_gru = nn.GRU(word_dim, hidden_dim, batch_first=True)

        # Identity baraye VQA loader compat (q_proj digar lazem nist)
        self.q_proj = nn.Identity()
        _ = question_dim

    def train(self, mode: bool = True) -> "QuestionDependentImageCaptioner":
        """Train caption layers; Faster R-CNN hamishe eval."""
        super().train(mode)
        self.region_encoder.train(mode)
        return self

    def _init_lstm_state(
        self, fused: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """h0/c0 az fused — soal + attended visual.

        Finglish: tanh(W(fused)) → state aval decoder QD-aware hast.
        """
        h = torch.tanh(self.init_h(fused))
        c = torch.tanh(self.init_c(fused))
        return h, c

    def _qctx(
        self,
        question_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Soal → qctx (N, hidden_dim) ba PAD-aware GRU.

        Finglish: question_ids lazeme; BOS/EOS/PAD toye length hesab mishe.
        """
        qe = self.q_emb(question_ids)
        lengths = (question_ids != self.question_pad_id).sum(dim=1).clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            qe, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.q_gru(packed)
        return h_n.squeeze(0)

    def _encode_regions(
        self,
        images: torch.Tensor,
        image_ids: Optional[torch.Tensor] = None,
        cache_dir: Optional[str] = None,
        save_cache: bool = True,
    ) -> torch.Tensor:
        """Faster R-CNN regions (bedoon GNN)."""
        return self.region_encoder(
            images,
            image_ids=image_ids,
            cache_dir=cache_dir,
            save_cache=save_cache,
        )

    def _encode_qd(
        self,
        images: torch.Tensor,
        question_ids: torch.Tensor,
        image_ids: Optional[torch.Tensor] = None,
        region_cache_dir: Optional[str] = None,
        save_region_cache: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Regions + qctx + fused (one-shot question attention + concat).

        Returns:
            regions: (N, R, 2048)
            fused: (N, H) — baraye LSTM init va har step
            qctx: (N, H) — baraye debug / VQA path
        """
        regions = self._encode_regions(
            images,
            image_ids=image_ids,
            cache_dir=region_cache_dir,
            save_cache=save_region_cache,
        )
        qctx = self._qctx(question_ids)
        attended = self.question_attention(regions, qctx)
        fused = self.fusion(qctx, attended)
        return regions, fused, qctx

    def _caption_step(
        self,
        regions: torch.Tensor,
        caption_tok: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
        fused: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Yek step: step_attention(h) + LSTM([word; fused; z_t]) + logits."""
        z_t = self.step_attention(regions, h)
        word = self.word_emb(caption_tok)
        h, c = self.lstm(torch.cat([word, fused, z_t], dim=-1), (h, c))
        out = self.dropout(h) + self.ctx_to_logit(z_t)
        return self.classifier(out), h, c

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
        """Predict caption_ids[:, 1:] — logits (N, T-1, V).

        Finglish — question_ids lazeme. scheduled_sampling_p:
            ba ehtemal p kalame pred; ba 1-p GT. p=0 → teacher forcing.
        """
        if question_ids is None:
            raise ValueError(
                "question_ids is required for QuestionDependentImageCaptioner"
            )
        regions, fused, _ = self._encode_qd(
            images,
            question_ids,
            image_ids=image_ids,
            region_cache_dir=region_cache_dir,
            save_region_cache=save_region_cache,
        )
        n = images.size(0)
        h, c = self._init_lstm_state(fused)
        logits: List[torch.Tensor] = []
        tok = caption_ids[:, 0]
        use_sampling = scheduled_sampling_p > 0.0
        for t in range(caption_ids.size(1) - 1):
            logit, h, c = self._caption_step(regions, tok, h, c, fused)
            logits.append(logit)
            if t < caption_ids.size(1) - 2:
                if use_sampling:
                    pred = logit.argmax(dim=-1).detach()
                    use_pred = (
                        torch.rand(n, device=images.device) < scheduled_sampling_p
                    )
                    tok = torch.where(use_pred, pred, caption_ids[:, t + 1])
                else:
                    tok = caption_ids[:, t + 1]
        return torch.stack(logits, dim=1)

    def _decode_caption(
        self,
        image: torch.Tensor,
        question_ids: Optional[torch.Tensor],
        max_len: int,
        collect_hidden: bool = False,
        image_ids: Optional[torch.Tensor] = None,
        region_cache_dir: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Greedy decode — baraye inference va differentiable VQA path."""
        if question_ids is None:
            raise ValueError(
                "question_ids is required for QuestionDependentImageCaptioner"
            )
        regions, fused, _ = self._encode_qd(
            image,
            question_ids,
            image_ids=image_ids,
            region_cache_dir=region_cache_dir,
            save_region_cache=True,
        )
        n = image.size(0)
        h, c = self._init_lstm_state(fused)
        tok = torch.full((n,), 1, dtype=torch.long, device=image.device)
        out = [tok]
        hidden_steps: List[torch.Tensor] = []
        for _ in range(max_len - 1):
            logit, h, c = self._caption_step(regions, tok, h, c, fused)
            if collect_hidden:
                hidden_steps.append(h)
            tok = logit.argmax(dim=-1)
            out.append(tok)
        cap = torch.stack(out, dim=1)
        if collect_hidden:
            return cap, hidden_steps
        return cap, None

    @staticmethod
    def _block_repeat_ngram(
        logp: torch.Tensor, seqs: torch.Tensor, ngram: int
    ) -> None:
        """Finglish — trigram blocking: n-gram tekrari → -inf roye logp."""
        rows, length = seqs.shape
        if length < ngram - 1:
            return
        prefix = seqs[:, -(ngram - 1) :]
        for r in range(rows):
            seq_r = seqs[r]
            pref_r = prefix[r]
            for i in range(length - ngram + 1):
                if torch.equal(seq_r[i : i + ngram - 1], pref_r):
                    logp[r, int(seq_r[i + ngram - 1])] = float("-inf")

    @torch.no_grad()
    def _beam_search(
        self,
        image: torch.Tensor,
        question_ids: Optional[torch.Tensor],
        max_len: int,
        beam_size: int = 5,
        length_alpha: float = 0.7,
        no_repeat_ngram: int = 3,
        image_ids: Optional[torch.Tensor] = None,
        region_cache_dir: Optional[str] = None,
        save_region_cache: bool = True,
    ) -> torch.Tensor:
        """Beam search decode — EOS-stop + length-norm + trigram block."""
        if question_ids is None:
            raise ValueError(
                "question_ids is required for QuestionDependentImageCaptioner"
            )
        bos_id, eos_id, pad_id = 1, 2, 0
        device = image.device
        regions, fused, _ = self._encode_qd(
            image,
            question_ids,
            image_ids=image_ids,
            region_cache_dir=region_cache_dir,
            save_region_cache=save_region_cache,
        )
        n = image.size(0)
        hidden = self.lstm_hidden
        beam = max(1, int(beam_size))

        h, c = self._init_lstm_state(fused)

        r_count, r_dim = regions.size(1), regions.size(2)
        regions = (
            regions.unsqueeze(1)
            .expand(n, beam, r_count, r_dim)
            .reshape(n * beam, r_count, r_dim)
        )
        fused = (
            fused.unsqueeze(1).expand(n, beam, hidden).reshape(n * beam, hidden)
        )
        h = (
            h.unsqueeze(1)
            .expand(n, beam, hidden)
            .reshape(n * beam, hidden)
            .contiguous()
        )
        c = (
            c.unsqueeze(1)
            .expand(n, beam, hidden)
            .reshape(n * beam, hidden)
            .contiguous()
        )

        beam_scores = torch.full((n, beam), float("-inf"), device=device)
        beam_scores[:, 0] = 0.0
        beam_scores = beam_scores.reshape(n * beam)
        tok = torch.full((n * beam,), bos_id, dtype=torch.long, device=device)
        seqs = tok.unsqueeze(1)
        finished = torch.zeros(n * beam, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            logit, h, c = self._caption_step(regions, tok, h, c, fused)
            logp = torch.log_softmax(logit, dim=-1)
            vocab = logp.size(-1)
            if no_repeat_ngram > 0:
                self._block_repeat_ngram(logp, seqs, no_repeat_ngram)
            if bool(finished.any()):
                logp[finished] = float("-inf")
                logp[finished, pad_id] = 0.0
            total = (beam_scores.unsqueeze(-1) + logp).view(n, beam * vocab)
            top_scores, top_idx = total.topk(beam, dim=-1)
            beam_id = torch.div(top_idx, vocab, rounding_mode="floor")
            token_id = top_idx % vocab
            offset = (torch.arange(n, device=device) * beam).unsqueeze(-1)
            flat = (beam_id + offset).reshape(n * beam)
            h, c = h[flat], c[flat]
            fused = fused[flat]
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
        """Inference ba beam search — soal lazeme."""
        return self._beam_search(
            image,
            question_ids,
            max_len,
            beam_size=beam_size,
            length_alpha=length_alpha,
            no_repeat_ngram=no_repeat_ngram,
            image_ids=image_ids,
            region_cache_dir=region_cache_dir,
            save_region_cache=save_region_cache,
        )

    def encode_caption(self, caption_ids: torch.Tensor) -> torch.Tensor:
        """Mean-pool token embeddings → v_cap (N, word_dim)."""
        return self.word_emb(caption_ids).mean(dim=1)

    def get_caption_embedding(
        self,
        image: torch.Tensor,
        question_ids: Optional[torch.Tensor] = None,
        differentiable: bool = False,
        image_ids: Optional[torch.Tensor] = None,
        region_cache_dir: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate caption + v_cap — VQAModel seda mizane.

        Eval: beam caption → mean word_emb.
        Train VQA (differentiable=True): mean LSTM hidden (EOS-aware) ta grad
        be q_emb beresad az tarigh fusion / question_attention.
        """
        if question_ids is None:
            raise ValueError(
                "question_ids is required for QuestionDependentImageCaptioner"
            )
        if differentiable:
            cap, hidden_steps = self._decode_caption(
                image,
                question_ids,
                max_len=20,
                collect_hidden=True,
                image_ids=image_ids,
                region_cache_dir=region_cache_dir,
            )
            assert hidden_steps is not None
            # Finglish: az avalin <eos> be baad mask — v_cap noise-e pad nadare
            hidden = torch.stack(hidden_steps, dim=1)
            gen = cap[:, 1:]
            eos = gen == 2
            prev_eos = eos.long().cumsum(dim=1) - eos.long()
            mask = (prev_eos == 0).to(hidden.dtype).unsqueeze(-1)
            v_cap = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            return v_cap, cap.detach()

        with torch.no_grad():
            cap = self.generate_caption(image, question_ids)
        return self.encode_caption(cap), cap
