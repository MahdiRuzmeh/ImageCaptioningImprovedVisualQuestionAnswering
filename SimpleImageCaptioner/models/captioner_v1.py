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

    def _cache_path(self, cache_dir: str, image_id: int) -> str:
        # keep it simple + stable (works on Kaggle/Windows)
        return str(Path(cache_dir) / f"{int(image_id)}.pt")

    def _load_cached(self, cache_dir: str, image_id: int, device: torch.device) -> Optional[torch.Tensor]:
        """
        Finglish:
        Inja region feature ha ro az disk load mikonim (per image_id).
        Hadaf: FasterRCNN har epoch dobare run nashe → time training kheili kam mishe.
        """
        try:
            p = self._cache_path(cache_dir, image_id)
            if not Path(p).exists():
                return None
            t = torch.load(p, map_location="cpu")
            if not isinstance(t, torch.Tensor):
                return None
            # expected shape: (max_regions, region_dim)
            if t.ndim != 2 or t.shape[0] != self.max_regions or t.shape[1] != self.region_dim:
                return None
            # AMP training may have saved fp16; model weights are fp32 at eval time.
            return t.float().to(device, non_blocking=(device.type == "cuda"))
        except Exception:
            return None

    def _save_cached(self, cache_dir: str, image_id: int, regions: torch.Tensor) -> None:
        """
        Finglish:
        Inja region tensor ro save mikonim. Save ro roye CPU anjam midim ta GPU memory/IO problem nashe.
        """
        try:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            p = self._cache_path(cache_dir, image_id)
            # always save fp32 CPU tensors (AMP may produce fp16 during training)
            torch.save(regions.detach().float().to("cpu"), p)
        except Exception:
            pass

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        image_ids: Optional[torch.Tensor] = None,
        cache_dir: Optional[str] = None,
        save_cache: bool = True,
    ) -> torch.Tensor:
        """(N,3,H,W) → (N, max_regions, region_dim).

        If ``cache_dir`` and ``image_ids`` are provided, region tensors are loaded/saved
        per image id. This avoids re-running Faster R-CNN every epoch.
        """
        device = images.device
        n = images.size(0)

        if cache_dir and image_ids is not None and image_ids.numel() == n:
            cached: List[Optional[torch.Tensor]] = [
                self._load_cached(cache_dir, int(image_ids[i].item()), device) for i in range(n)
            ]
            if all(t is not None for t in cached):
                return torch.stack([t for t in cached if t is not None], dim=0)

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

        regions = self.roi_to_region(torch.stack(batch_roi, dim=0))

        if cache_dir and image_ids is not None and image_ids.numel() == n and save_cache:
            # write per-sample so partial cache hits still help later
            for i in range(n):
                self._save_cached(cache_dir, int(image_ids[i].item()), regions[i])

        return regions


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
    VQA: soal (question_ids) az ``q_emb`` joda az ``word_emb`` encode mishan ta
    caption vocabulary va question vocabulary mixed nashan.

    Do vocabulary joda:
        - ``word_emb`` + ``classifier`` → token haye **caption** (train roye MSCOCO)
        - ``q_emb`` → token haye **soal** VQA (faghat vaghti ``question_vocab_size`` pass shode)
        - dar VQA train: faghat ``q_emb`` + ``q_proj`` update mishan (indirect az answer loss)

    Init signature hamoon ImageCaptionerV1 hast ta ``captioner_adapter`` kar kone.
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
    ) -> None:
        """Caption decoder + optional question embedding baraye VQA.

        Args:
            vocab_size: size vocabulary **caption** (``word_emb``, ``classifier``).
            pad_id: index PAD baraye caption tokens.
            question_vocab_size: age set shavad, ``q_emb`` sakhte mishavad baraye soal VQA.
                Dar train caption-only (``SimpleImageCaptioner/train.py``) pass nemishe.
            question_pad_id: PAD baraye ``q_emb``; default hamoon ``pad_id``.
        """
        super().__init__()
        self.lstm_hidden = hidden_dim
        self.embed_dim = embed_dim if embed_dim is not None else hidden_dim
        self.word_dim = word_dim

        self.region_encoder = RegionEncoder(max_regions, region_dim)

        self.attention = RegionAttention(region_dim, hidden_dim, self.embed_dim)

        self.word_emb = nn.Embedding(vocab_size, word_dim, padding_idx=pad_id)

        self.lstm = nn.LSTMCell(word_dim + self.embed_dim, hidden_dim)

        # Finglish — dropout (paper §5, p=0.5):
        #   Roye hidden LSTM ghabl az classifier; faghat train mode tasir dare.
        #   Mesal h=[0.2,-0.1,...] → ba p=0.5 nesfe element ha 0 mishan, baghi ×2.
        #   Inference (eval) dropout off hast — hidden kamel be classifier mire.
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, vocab_size)

        # embedding joda baraye soal VQA — ba word_emb caption share nemikone
        self.q_emb: Optional[nn.Embedding] = None
        if question_vocab_size is not None:
            q_pad = pad_id if question_pad_id is None else question_pad_id
            self.q_emb = nn.Embedding(
                question_vocab_size, word_dim, padding_idx=q_pad
            )

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
        VQA: mean-pool ``q_emb(question_ids)`` → ``q_proj`` → be ``h_{t-1}`` ezafe
        mishavad ta caption **question-guided** beshe (na caption omumi).

        ``word_emb`` faghat baraye token haye caption dar decode estefade mishe.
        """
        if question_ids is None:
            return torch.zeros(batch, self.lstm_hidden, device=device)
        if self.q_emb is None:
            raise RuntimeError(
                "question_ids provided but captioner has no q_emb; "
                "pass question_vocab_size when loading for VQA."
            )
        qe = self.q_emb(question_ids).mean(dim=1)
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

        h, c = self.lstm(torch.cat([word, attended], dim=-1), (h, c))
        return self.classifier(self.dropout(h)), h, c

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

        Finglish — ``scheduled_sampling_p`` (paper §5, mesal p=0.24):
            Har step ba ehtemal p kalame ghabli **model** (argmax) be decoder mire;
            ba ehtemal 1-p hamoon **GT** (`caption_ids[:, t+1]`).
            Mesal GT="a dog sits": step2 pred="cat" → age rand<p input step3="cat", else "dog".
            p=0 → faghat teacher forcing; validation hamishe p=0.
        """
        regions = self.region_encoder(
            images,
            image_ids=image_ids,
            cache_dir=region_cache_dir,
            save_cache=save_region_cache,
        )
        n = images.size(0)

        # toye train question haro nemidim behesh.
        qctx = self._qctx(question_ids, n, images.device)
        h = torch.zeros(n, self.lstm_hidden, device=images.device)
        c = torch.zeros_like(h)
        logits: List[torch.Tensor] = []
        tok = caption_ids[:, 0]
        use_sampling = scheduled_sampling_p > 0.0
        for t in range(caption_ids.size(1) - 1):
            logit, h, c = self._caption_step(regions, tok, h, c, qctx)
            logits.append(logit)
            if t < caption_ids.size(1) - 2:
                if use_sampling:
                    pred = logit.argmax(dim=-1).detach()
                    use_pred = torch.rand(n, device=images.device) < scheduled_sampling_p
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
        """Loop greedy decode — moshtarak baraye inference va train VQA.

        Har step:
            qctx = f(q_emb, question_ids)  → attention(regions, h + qctx)
            → LSTM → logits → argmax token

        Args:
            collect_hidden: age ``True``, hidden state LSTM har step save mishe.
                baraye ``get_caption_embedding(differentiable=True)`` lazem ast
                ta v_cap be q_emb gradient bede (bedoon in, argmax gradient ro mibarad).

        Returns:
            cap: token ids caption tolid shode (N, max_len)
            hidden_steps: list hidden ha ya ``None``
        """
        regions = self.region_encoder(image, image_ids=image_ids, cache_dir=region_cache_dir)
        n = image.size(0)
        qctx = self._qctx(question_ids, n, image.device)
        h = torch.zeros(n, self.lstm_hidden, device=image.device)
        c = torch.zeros_like(h)
        tok = torch.full((n,), 1, dtype=torch.long, device=image.device)
        out = [tok]
        hidden_steps: List[torch.Tensor] = []
        for _ in range(max_len - 1):
            logit, h, c = self._caption_step(regions, tok, h, c, qctx)
            if collect_hidden:
                hidden_steps.append(h)
            tok = logit.argmax(dim=-1)
            out.append(tok)
        cap = torch.stack(out, dim=1)
        if collect_hidden:
            return cap, hidden_steps
        return cap, None

    @torch.no_grad()
    def generate_caption(
        self,
        image: torch.Tensor,
        question_ids: Optional[torch.Tensor] = None,
        max_len: int = 20,
    ) -> torch.Tensor:
        """Greedy decode az BOS (id=1) — baraye inference va eval VQA.

        ``@torch.no_grad()``: inference faghat; train VQA az ``_decode_caption`` ba
        ``collect_hidden=True`` estefade mikone ta grad dashte bashe.
        """
        cap, _ = self._decode_caption(image, question_ids, max_len, collect_hidden=False)
        return cap

    def encode_caption(self, caption_ids: torch.Tensor) -> torch.Tensor:
        """Mean-pool token embeddings → v_cap (N, word_dim) — paper §3.4."""
        return self.word_emb(caption_ids).mean(dim=1)

    def get_caption_embedding(
        self,
        image: torch.Tensor,
        question_ids: Optional[torch.Tensor] = None,
        differentiable: bool = False,
        image_ids: Optional[torch.Tensor] = None,
        region_cache_dir: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate caption + v_cap — in method ro VQAModel seda mizane.

        Do halat:

        **Eval / inference** (``differentiable=False``):
            - ``generate_caption`` ba ``no_grad``
            - v_cap = mean-pool ``word_emb`` roye token haye caption (paper §3.4)
            - in hamoon chizi hast ke maghale tozih mide

        **Train VQA** (``differentiable=True``):
            - caption GT baraye (image, question) nadarim → captioner ro direct train nemikonim
            - argmax token gradient ro cut mikone → nemitoonim ``word_emb(cap)`` ro baraye backprop estefade konim
            - hal: v_cap = mean(LSTM hidden states) dar hamin decode loop
            - grad path: answer_loss → v_cap → h → attention(h+qctx) → qctx → q_emb
            - faghat ``q_emb`` (+ ``q_proj``) trainable hastan; LSTM/attention frozen vali grad az input rad mishe

        Returns:
            v_cap: (N, word_dim) — caption representation baraye fuse ba v_att
            cap: token ids — dar train ``detach`` shode (faghat baraye log/debug)
        """
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
            v_cap = torch.stack(hidden_steps, dim=1).mean(dim=1)
            return v_cap, cap.detach()

        with torch.no_grad():
            cap = self.generate_caption(image, question_ids)
        return self.encode_caption(cap), cap
