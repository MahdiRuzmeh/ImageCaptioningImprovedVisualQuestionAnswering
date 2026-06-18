"""Simple captioner ba region attention — paper §3.3 + VQA hook §3.4.

Architecture (paper):
- Region features v_i ∈ R^2048 az Faster R-CNN (frozen) -> hyperparametr haye magale (32*2048)
- Har step t: α_{ti} = softmax(f_att(v_i, m_{t-1} + q_bias)) -> yani toye har step LSTM ahamiyat har region ro hesab mikonim. 
    dar training similarity(region_i,caption(t-1)) ro hesab mikonim (chon ba question nadarim toye in marhale. question=[0])
    vali vagti model VQA ro mikhaym train konim(Image caotioner ro fine tune konim) miyaym shebahat (caption+question) ro roye 
    region ha hesab mikonim.
- z_t = Σ α_{ti} v_i → project be 512 → concat ba word embedding → LSTM
 z_t hamon caption_related_img_feat hast. ke [32]*[32*2048] = [2048] -> midim be FC layer -> [2048] -> [512] -> concat mikonim ba h(t-1) 
 va ye vector 1024 tayi ro midim be LSTM.
- v_cap = mean-pool embedding haye caption tolid shode
"""

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

        # inja migim feature extractor ro freeze kon.
        for p in self.detector.parameters():
            p.requires_grad = False

        self.detector.eval()

        # inja ham ba ye linier layer khorouji ro be abaadi ke delemon mikhad tabdil mikonim.
        self.roi_to_region = nn.Linear(self.ROI_FEAT_DIM, region_dim)

    def train(self, mode: bool = True) -> "RegionEncoder":
        """Detector hamishe eval mimune (BN freeze)."""
        super().train(mode)
        self.detector.eval()
        return self

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """(N,3,H,W) → (N, max_regions, region_dim)."""
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
        batch_regions: List[torch.Tensor] = []
        for chunk in chunks:
            r = chunk[: self.max_regions]
            if r.size(0) < self.max_regions:
                pad = torch.zeros(
                    (self.max_regions - r.size(0), r.size(1)), device=r.device
                )
                r = torch.cat([r, pad], dim=0)
            batch_regions.append(r)
        return self.roi_to_region(torch.stack(batch_regions, dim=0))


class RegionAttention(nn.Module):
    """Soft attention rooye region-ha — paper eq. 5–7, §5 (512-D working space).

    h_{t-1} va har v_i ro be 512 project mikonim, score = dot product,
    softmax → z_t dar 2048-D, bad ctx_proj → 512 baraye LSTM.
    """

    def __init__(
        self,
        region_dim: int,
        lstm_hidden: int,
        embed_dim: int = 512,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.h_proj = nn.Linear(lstm_hidden, embed_dim)
        self.v_proj = nn.Linear(region_dim, embed_dim)
        self.ctx_proj = nn.Linear(region_dim, embed_dim)

    def forward(
        self, regions: torch.Tensor, h_prev: torch.Tensor
    ) -> torch.Tensor:
        """regions (N,R,2048), h_prev (N,512) → context (N,512)."""
        h_att = self.h_proj(h_prev)
        v_att = self.v_proj(regions)
        scores = (v_att * h_att.unsqueeze(1)).sum(dim=-1)
        weights = torch.softmax(scores, dim=-1)
        z = torch.einsum("br,brd->bd", weights, regions)
        return self.ctx_proj(z)


class SimpleImageCaptioner(BaseImageCaptioner):
    """Captioner sade ba per-step region attention — VQA-ready.

    Paper §3.3: attention dar har timestep ba m_{t-1}.
    VQA: soal (question_ids) be hidden state ezafe mishe ta caption
    question-conditioned beshe (mesl forward VQAModel ke q_ids pass mide).

    Init signature hamoon ImageCaptionerV1 hast ta `captioner_adapter` kar kone.
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
    ) -> None:
        super().__init__()
        self.lstm_hidden = hidden_dim
        self.embed_dim = embed_dim if embed_dim is not None else hidden_dim
        self.word_dim = word_dim

        self.region_encoder = RegionEncoder(max_regions, region_dim)

        self.attention = RegionAttention(region_dim, hidden_dim, self.embed_dim)

        self.word_emb = nn.Embedding(vocab_size, word_dim, padding_idx=pad_id)

        self.lstm = nn.LSTMCell(word_dim + self.embed_dim, hidden_dim)

        self.classifier = nn.Linear(hidden_dim, vocab_size)

        # agar word_dim != hidden_dim, soal ro project mikonim (VQA compat)
        self.q_proj = (
            nn.Linear(word_dim, hidden_dim)
            if word_dim != hidden_dim
            else nn.Identity()
        )
        _ = question_dim  # baraye YAML parity ba VQA; pool shode word_dim hast

    def train(self, mode: bool = True) -> "SimpleImageCaptioner":
        """Train caption layers; Faster R-CNN hamishe eval."""
        super().train(mode)
        self.region_encoder.train(mode)
        return self

    def _qctx(
        self,
        question_ids: Optional[torch.Tensor],
        batch: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Soal ro be vector bias tabdil mikone baraye attention.

        Agar question_ids=None (caption-only train), zero vector bar migardune.
        VQA: mean-pool embedding soal → q_proj → ezafe be h_{t-1}.
        """
        if question_ids is None:
            return torch.zeros(batch, self.lstm_hidden, device=device)
        qe = self.word_emb(question_ids).mean(dim=1)
        return self.q_proj(qe)

    def _caption_step(
        self,
        regions: torch.Tensor,
        caption_tok: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
        qctx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Yek step decode: attention + LSTM + logits."""
        # toye inja question vector va caption vector baham jaam mishan va
        # be onvan query be img feature zade mishan. 
        # attention query vector dimention= [N* 512] (natije attention be ezaye har tasvir ye vector 512d hast.)
        # img feature dimention [N* 32* 2048] hast.
        # baraye mohasebe similarity bayad project konim be space ba dimention
        # [N* 32 * 512]. alan mishe similarity hesab kard.
        # similarity([32* 512], [512])= [32]
        # result attention dimention= [N* 32]
        attended = self.attention(regions, h + qctx)
        
        word = self.word_emb(caption_tok)

        # ba ravesh teacher forcing train mishe Image captioner.
        # yani caption word gound truth ro behesh midim toye train.
        h, c = self.lstm(torch.cat([word, attended], dim=-1), (h, c))
        return self.classifier(h), h, c

    def forward_train(
        self,
        images: torch.Tensor,
        caption_ids: torch.Tensor,
        question_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Teacher forcing: predict caption_ids[:, 1:] — logits (N, T-1, V)."""
        regions = self.region_encoder(images)
        n = images.size(0)
        
        # toye train question haro nemidim behesh.
        qctx = self._qctx(question_ids, n, images.device)
        h = torch.zeros(n, self.lstm_hidden, device=images.device)
        c = torch.zeros_like(h)
        logits: List[torch.Tensor] = []
        for t in range(caption_ids.size(1) - 1):
            logit, h, c = self._caption_step(
                regions, caption_ids[:, t], h, c, qctx
            )
            logits.append(logit)
        return torch.stack(logits, dim=1)

    @torch.no_grad()
    def generate_caption(
        self,
        image: torch.Tensor,
        question_ids: Optional[torch.Tensor] = None,
        max_len: int = 20,
    ) -> torch.Tensor:
        """Greedy decode az BOS (id=1) — baraye inference va VQA."""
        regions = self.region_encoder(image)
        n = image.size(0)

        # toye inja question vector va caption vector baham jaam mishan va (toye train Image captioner inja question ro nadaram)
        # dar nahayat question_caption_context be onvan query be img region features zade mishan. 
        # img feature dimention [N* 32* 2048] hast.
        # baraye mohasebe similarity bayad project konim be space ba dimention
        # [N* 32 * 512]. alan mishe similarity hesab kard. (chon h(t-1) ya hamon caption_feature vector 512d hast.)
        # similarity([32* 512], [512])= [32]
        # result attention dimention= [N* 32]
        qctx = self._qctx(question_ids, n, image.device)

        h = torch.zeros(n, self.lstm_hidden, device=image.device)
        c = torch.zeros_like(h)
        tok = torch.full((n,), 1, dtype=torch.long, device=image.device)
        out = [tok]
        for _ in range(max_len - 1):
            logit, h, c = self._caption_step(regions, tok, h, c, qctx)
            tok = logit.argmax(dim=-1)
            out.append(tok)
        return torch.stack(out, dim=1)

    def encode_caption(self, caption_ids: torch.Tensor) -> torch.Tensor:
        """Mean-pool token embeddings → v_cap (N, word_dim) — paper §3.4."""
        return self.word_emb(caption_ids).mean(dim=1)

    @torch.no_grad()
    def get_caption_embedding(
        self,
        image: torch.Tensor,
        question_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate caption + v_cap — in method ro VQAModel seda mizane."""
        cap = self.generate_caption(image, question_ids)
        return self.encode_caption(cap), cap
