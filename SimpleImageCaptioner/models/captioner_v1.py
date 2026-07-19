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
- Region-ha ghabl az attention az RelationGNN migozarand (ta betonim ertebat beyne object-ha ro peyda konim).
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
from .relation_gnn import RelationGNN


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

        # Finglish — double-normalization fix:
        #   Dataset ghablan image ro ba ImageNet mean/std normalize mikone.
        #   GeneralizedRCNNTransform default dobare hamon normalize ro mizane → double!
        #   mean=0,std=1 mizarim ta detector faghat resize kone (na renormalize).
        #   Bad az in fix region_cache ghadi ghalat hast → cache ro pak kon + retrain.
        self.detector.transform.image_mean = [0.0, 0.0, 0.0]
        self.detector.transform.image_std = [1.0, 1.0, 1.0]

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
        Finglish — Bug 2 fix: cache hala raw ROI (1024D) negahdari mikone, na post-roi_to_region.
        Shape check ba ROI_FEAT_DIM anjam mishe (na region_dim=2048).
        In yani roi_to_region har bar az cache load mishe va gradient migire (Bug 1 fix).
        Cache haye ghadi (2048D) auto-invalidate mishan → dar epoch 1 dobare compute mishan.
        """
        try:
            p = self._cache_path(cache_dir, image_id)
            if not Path(p).exists():
                return None
            t = torch.load(p, map_location="cpu")
            if not isinstance(t, torch.Tensor):
                return None
            # expected shape: (max_regions, ROI_FEAT_DIM=1024) — raw backbone output
            if t.ndim != 2 or t.shape[0] != self.max_regions or t.shape[1] != self.ROI_FEAT_DIM:
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

    def forward(
        self,
        images: torch.Tensor,
        image_ids: Optional[torch.Tensor] = None,
        cache_dir: Optional[str] = None,
        save_cache: bool = True,
    ) -> torch.Tensor:
        """(N,3,H,W) → (N, max_regions, region_dim).

        Finglish — Bug 1 + Bug 2 fix: @torch.no_grad() az kol forward bardashte shod.
        Hala faghat backbone (Faster R-CNN) zir with torch.no_grad() ejra mishe.(yani freeze shode mimone)
        roi_to_region birun az no_grad hast → gradient migire → train mishe.(yani layer FC ke 1024d be 2048d tabdil mikone ham train mishe)
        Cache hala raw ROI (1024D) negahdari mikone; roi_to_region baad az load ejra mishe.
        """
        device = images.device
        n = images.size(0)

        # --- cache hit: load raw ROI (1024D), then apply trainable roi_to_region ---
        if cache_dir and image_ids is not None and image_ids.numel() == n:
            cached: List[Optional[torch.Tensor]] = [
                self._load_cached(cache_dir, int(image_ids[i].item()), device) for i in range(n)
            ]
            if all(t is not None for t in cached):
                raw = torch.stack([t for t in cached if t is not None], dim=0)
                return self.roi_to_region(raw)

        # --- cache miss: run frozen Faster R-CNN backbone under no_grad ---
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
            raw_roi = torch.stack(batch_roi, dim=0)  # (N, max_regions, 1024) — no grad

        # save raw 1024D ROI to cache (before roi_to_region) so cache stays valid
        # across epochs even as roi_to_region weights change during training.
        if cache_dir and image_ids is not None and image_ids.numel() == n and save_cache:
            for i in range(n):
                self._save_cached(cache_dir, int(image_ids[i].item()), raw_roi[i])

        # apply trainable projection outside no_grad → receives gradients every forward pass
        return self.roi_to_region(raw_roi)


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
    """Captioner sade ba per-step region attention — VQA-ready + QD train.

    Paper §3.3: attention dar har timestep ba m_{t-1}.
    QD / VQA: soal az ``q_emb`` + ``q_gru`` encode mishe (joda az ``word_emb``).

    Do vocabulary joda:
        - ``word_emb`` + ``classifier`` → token haye **caption**
        - ``q_emb`` + ``q_gru`` → token haye **soal** (vaghti ``question_vocab_size`` set shode)

    Conditioning:
        - ``qctx`` → attention query: ``attn_query_proj([h; qctx])``
        - ``qctx`` → LSTM input: ``[word; attended; qctx]``
        - age ``question_ids=None`` → ``qctx=0`` (backward compat MSCOCO-only)
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
        use_gnn: bool = True,
        gnn_dim: Optional[int] = None,
    ) -> None:
        """Caption decoder + optional GRU question encoder baraye QD / VQA.

        Args:
            vocab_size: size vocabulary **caption** (``word_emb``, ``classifier``).
            pad_id: index PAD baraye caption tokens.
            question_vocab_size: age set shavad, ``q_emb`` + ``q_gru`` sakhte mishavad.
            question_pad_id: PAD baraye ``q_emb``; default hamoon ``pad_id``.
        """
        super().__init__()
        self.lstm_hidden = hidden_dim
        self.embed_dim = embed_dim if embed_dim is not None else hidden_dim
        self.word_dim = word_dim
        self.region_dim = region_dim
        self.use_gnn = use_gnn
        self.question_pad_id = pad_id if question_pad_id is None else int(question_pad_id)

        self.region_encoder = RegionEncoder(max_regions, region_dim)

        gnn_work_dim = gnn_dim if gnn_dim is not None else self.embed_dim
        # Finglish — GNN roye region 2048→512, message-pass, 512→2048.
        # Mesal: node «dog» message az node «ball» migire → caption «dog chasing ball» ro dara nahayat generate mikone.
        self.gnn_in = nn.Linear(region_dim, gnn_work_dim)
        self.gnn = RelationGNN(gnn_work_dim)
        self.gnn_out = nn.Linear(gnn_work_dim, region_dim)

        self.attention = RegionAttention(region_dim, hidden_dim, self.embed_dim)

        # Finglish — attn query = Linear(concat(h, qctx)):
        #   sum(h+qctx) yek W moshtarak dare → feature ha cancel mishan.
        #   concat → W_h@h + W_q@qctx joda → QD attention ghavitar.
        #   qctx=0 (COCO) → [h; 0] hanuz OK.
        self.attn_query_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        self.word_emb = nn.Embedding(vocab_size, word_dim, padding_idx=pad_id)

        # Finglish — LSTM input = word + attended context + qctx (QD strong conditioning).
        # Age soal nabashe qctx=0 → behave mesl ghabl, ama input size bozorgtar ast.
        self.lstm = nn.LSTMCell(
            word_dim + self.embed_dim + hidden_dim, hidden_dim
        )

        # Finglish — dropout (paper §5, p=0.5):
        #   Roye hidden LSTM ghabl az classifier; faghat train mode tasir dare.
        #   Mesal h=[0.2,-0.1,...] → ba p=0.5 nesfe element ha 0 mishan, baghi ×2.
        #   Inference (eval) dropout off hast — hidden kamel be classifier mire.
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, vocab_size)

        # Finglish — deep-output grounding (Show-Attend-Tell eq. 7):
        #   logit faghat az h LSTM nemiyad; attended (image context) va word emb
        #   mostaghim be classifier ezafe mishan → model nemitone image ro ignore kone.
        #   ctx_to_logit: context(512)→hidden, word_to_logit: word(512)→hidden.
        self.ctx_to_logit = nn.Linear(self.embed_dim, hidden_dim)
        self.word_to_logit = nn.Linear(word_dim, hidden_dim)

        # Finglish — Bug 3 fix: LSTM state az mean region initialize mishe (na zeros).
        # Ghabl h=0 bood → attention dar step 0 koor bood → "a man" mode collapse.(yani hame caption ha avaleshon yeksan shoro mishod ke eshtebah bood)
        # Hala: h = tanh(W * mean(regions)), c = tanh(W * mean(regions)) — image-specific.
        # Referens: Xu et al. 2015 "Show, Attend and Tell" — hamoon technique.
        self.region_init_h = nn.Linear(region_dim, hidden_dim)
        self.region_init_c = nn.Linear(region_dim, hidden_dim)

        # embedding + GRU joda baraye soal — ba word_emb caption share nemikone
        self.q_emb: Optional[nn.Embedding] = None
        self.q_gru: Optional[nn.GRU] = None
        if question_vocab_size is not None:
            self.q_emb = nn.Embedding(
                question_vocab_size, word_dim, padding_idx=self.question_pad_id
            )
            # Finglish — GRU soal: sequence token → last hidden = qctx (hidden_dim).
            # behtar az mean-pool: tartib kalamat soal hefz mishe.
            self.q_gru = nn.GRU(
                word_dim, hidden_dim, batch_first=True
            )

        # q_proj digar lazem nist (GRU mostaghim hidden_dim mide); Identity baraye compat
        self.q_proj = nn.Identity()
        _ = question_dim  # baraye YAML parity ba VQA

    def train(self, mode: bool = True) -> "SimpleImageCaptioner":
        """Train caption layers; Faster R-CNN hamishe eval."""
        super().train(mode)
        self.region_encoder.train(mode)
        return self

    def _init_lstm_state(
        self, regions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """LSTM hidden va cell state ro az mean region features initialize mikone.

        Finglish — Bug 3 fix: be jaye zeros, h va c az mean-pool region compute mishan.
        mean(regions) → (N, region_dim) → Linear → tanh → (N, hidden_dim).
        In yani step 0 attention ya query mokhtas har tasvir hast, na zero (koor).
        Referens: Xu et al. 2015 "Show, Attend and Tell", section 3.1.
        """
        mean_r = regions.mean(dim=1)
        h = torch.tanh(self.region_init_h(mean_r))
        c = torch.tanh(self.region_init_c(mean_r))
        return h, c

    def _qctx(
        self,
        question_ids: Optional[torch.Tensor],
        batch: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Soal ro ba GRU encode mikone → qctx (N, hidden_dim).

        Agar question_ids=None, zero vector (caption omumi / backward compat).
        Agar soal dare: ``q_emb`` → ``q_gru`` (PAD-aware last state) → qctx.

        qctx ham be attention (concat ba h → proj) mire, ham be LSTM input concat.
        """
        if question_ids is None:
            return torch.zeros(batch, self.lstm_hidden, device=device)
        if self.q_emb is None or self.q_gru is None:
            raise RuntimeError(
                "question_ids provided but captioner has no q_emb/q_gru; "
                "pass question_vocab_size when building the model."
            )
        # (N, T, word_dim)
        qe = self.q_emb(question_ids)
        # mask PAD: length >= 1 baraye har sample (khali → 1)
        lengths = (question_ids != self.question_pad_id).sum(dim=1).clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            qe, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.q_gru(packed)
        # h_n: (1, N, hidden) → (N, hidden)
        return h_n.squeeze(0)

    def _encode_regions(
        self,
        images: torch.Tensor,
        image_ids: Optional[torch.Tensor] = None,
        cache_dir: Optional[str] = None,
        save_cache: bool = True,
    ) -> torch.Tensor:
        """Faster R-CNN regions; age ``use_gnn`` → RelationGNN (paper §3.1, §3.3).

        Finglish — Bug 4 fix (second part): residual connection be GNN path ezafe shod.
        Ghabl: regions be kolli ba GNN output replace mishodan → age GNN bad bood, kharab.
        Hala: regions + gnn_delta → feature asli hifz mishe + GNN context ezafe mishe.
        """
        regions = self.region_encoder(
            images,
            image_ids=image_ids,
            cache_dir=cache_dir,
            save_cache=save_cache,
        )
        if not self.use_gnn:
            return regions
        gnn_delta = self.gnn_out(self.gnn(self.gnn_in(regions)))
        return regions + gnn_delta

    def _caption_step(
        self,
        regions: torch.Tensor,
        caption_tok: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
        qctx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Yek step decode: attention(concat+proj) + LSTM([word; attended; qctx]) + logits."""
        # Finglish — strong QD conditioning:
        #   1) query = attn_query_proj([h; qctx]) → region-haye related be soal
        #   2) LSTM input = [word; attended; qctx] → soal mostaghim toye decode
        attn_query = self.attn_query_proj(torch.cat([h, qctx], dim=-1))
        attended = self.attention(regions, attn_query)
        word = self.word_emb(caption_tok)
        h, c = self.lstm(torch.cat([word, attended, qctx], dim=-1), (h, c))
        # Finglish — deep-output: logit = f(h, attended, word).
        # image context (attended) mostaghim vared prediction mishe → grounding ejbari.
        out = self.dropout(h) + self.ctx_to_logit(attended) + self.word_to_logit(word)
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

        Finglish — ``scheduled_sampling_p`` (paper §5, mesal p=0.24):
            Har step ba ehtemal p kalame ghabli **model** (argmax) be decoder mire;
            ba ehtemal 1-p hamoon **GT** (`caption_ids[:, t+1]`).
            Mesal GT="a dog sits": step2 pred="cat" → age rand<p input step3="cat", else "dog".
            p=0 → faghat teacher forcing; validation hamishe p=0.
        """
        regions = self._encode_regions(
            images,
            image_ids=image_ids,
            cache_dir=region_cache_dir,
            save_cache=save_region_cache,
        )
        n = images.size(0)

        # toye train question haro nemidim behesh.
        qctx = self._qctx(question_ids, n, images.device)
        # Bug 3 fix: h,c az mean region initialize mishan (na zeros) — image-specific start.
        h, c = self._init_lstm_state(regions)
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
            qctx = f(q_emb, question_ids)  → attention(regions, proj([h; qctx]))
            → LSTM → logits → argmax token

        Args:
            collect_hidden: age ``True``, hidden state LSTM har step save mishe.
                baraye ``get_caption_embedding(differentiable=True)`` lazem ast
                ta v_cap be q_emb gradient bede (bedoon in, argmax gradient ro mibarad).

        Returns:
            cap: token ids caption tolid shode (N, max_len)
            hidden_steps: list hidden ha ya ``None``
        """
        regions = self._encode_regions(
            image,
            image_ids=image_ids,
            cache_dir=region_cache_dir,
            save_cache=True,
        )
        n = image.size(0)
        qctx = self._qctx(question_ids, n, image.device)
        # Bug 3 fix: h,c az mean region initialize mishan (na zeros) — image-specific start.
        h, c = self._init_lstm_state(regions)
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

    @staticmethod
    def _block_repeat_ngram(
        logp: torch.Tensor, seqs: torch.Tensor, ngram: int
    ) -> None:
        """Finglish — trigram blocking (in-place roye log-prob):
            Age yek n-gram (mesal 3-tayi) ghablan tooye hamin seq oomade bashe,
            token-i ke oon n-gram ro tekrar mikone -inf mishe → loop hazf mishe.
            Mesal: seq=[...,a,horse,a] + ngram=3 → token «horse» block (chon «a horse» tekrari).
        """
        rows, length = seqs.shape
        if length < ngram - 1:
            return
        prefix = seqs[:, -(ngram - 1):]
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
        """Beam search decode — caption behtar az greedy tolid mikone.

        Finglish — beam search chiye?
            Greedy: har step FAGHAT 1 kalame (argmax) entekhab mikone → age ye kalame
            eshtebah bashe, dige nemitone jobran kone (caption kharab mishe).
            Beam search: har step ``beam_size`` ta behtarin "masir" (hypothesis) ro
            hamzaman negah midare va edame mide → ehtemal peyda kardane caption behtar bishtar.

        Mesal (beam_size=2), score = jam log-probability har kalame:
            step1 (bad az <bos>):  kandid ha → "a" (-0.2),  "the" (-0.9)   → 2 ta top negah dashte mishan
            step2:  "a man" (-0.5), "a dog" (-1.1), "the man" (-1.3), ...   → baz 2 ta top: «a man», «a dog»
            step3:  «a man riding» (-0.8), «a dog running» (-1.4), ...      → va hamintor ta EOS ya max_len
            akhar: az beyne hame hypothesis ha, behtarin (bad az length-norm) entekhab mishe.

        Se behbood ke ezafe shode (har kodom yek bug-e greedy ro hal mikone):
            1) EOS-stop: hypothesis ke <eos> bezane "tamum" mishe (freeze) → dige kalame
               ezafe nemizane → caption mesl «...a table a table a table» nemishe.
            2) length-norm: score ro bar ``length^length_alpha`` taghsim mikonim. chera?
               chon jam log-prob baraye jomle boland hamishe manfi-tar (badtar) mishe →
               bedoon in, model jomle haye kheili kootah ro tarjih mide. alpha=0.7 motavaset.
            3) trigram block (``no_repeat_ngram``): har 3-gram faghat 1 bar → loop hazf.

        Args:
            image: (N,3,H,W) — mitone batch bashe (har image joda beam khodesho dare).
            question_ids: baraye VQA question-guided; None → caption omumi.
            max_len: max tedad token.
            beam_size: tedad masir hamzaman (5 default; 1 → greedy ama ba EOS-stop+block).
            length_alpha: sheddat jarime tool (0=bi-asar, bozorgtar=jomle bolandtar tarjih).
            no_repeat_ngram: tool n-gram ke nabayad tekrar she (3 = trigram).

        Returns:
            (N, L) token ids — baraye har image behtarin caption (ba <bos>, ta <eos>/max_len).
        """
        bos_id, eos_id, pad_id = 1, 2, 0
        device = image.device
        regions = self._encode_regions(
            image,
            image_ids=image_ids,
            cache_dir=region_cache_dir,
            save_cache=save_region_cache,
        )
        n = image.size(0)
        hidden = self.lstm_hidden
        beam = max(1, int(beam_size))

        qctx = self._qctx(question_ids, n, device)
        h, c = self._init_lstm_state(regions)

        r_count, r_dim = regions.size(1), regions.size(2)
        regions = regions.unsqueeze(1).expand(n, beam, r_count, r_dim).reshape(n * beam, r_count, r_dim)
        qctx = qctx.unsqueeze(1).expand(n, beam, hidden).reshape(n * beam, hidden)
        h = h.unsqueeze(1).expand(n, beam, hidden).reshape(n * beam, hidden).contiguous()
        c = c.unsqueeze(1).expand(n, beam, hidden).reshape(n * beam, hidden).contiguous()

        # faghat beam 0 active-e ta step aval B token motafavet bede (na B beam yeksan)
        beam_scores = torch.full((n, beam), float("-inf"), device=device)
        beam_scores[:, 0] = 0.0
        beam_scores = beam_scores.reshape(n * beam)
        tok = torch.full((n * beam,), bos_id, dtype=torch.long, device=device)
        seqs = tok.unsqueeze(1)
        finished = torch.zeros(n * beam, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            logit, h, c = self._caption_step(regions, tok, h, c, qctx)
            logp = torch.log_softmax(logit, dim=-1)
            vocab = logp.size(-1)
            if no_repeat_ngram > 0:
                self._block_repeat_ngram(logp, seqs, no_repeat_ngram)
            # hypothesis haye tamum shode freeze: faghat PAD ba score 0 (taghir nemikone)
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
        """Finglish — inference decode ba beam search (default beam=5):
            EOS-stop + length-norm + trigram block → caption tamiz, bedoon tekrar.
            beam_size=1 → greedy (ama baz ham EOS-stop + trigram block dare).
            train VQA az in estefade nemikone (oon ``_decode_caption`` ba grad dare).
            Mesal: «a small plane flying through the sky <eos>».
        """
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
            - grad path: answer_loss → v_cap → h → attn_query_proj([h;qctx]) → qctx → q_emb
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
            # ----------------------------------------------------------
            # EOS-aware masked mean pooling for v_cap
            # ----------------------------------------------------------
            # EN: `_decode_caption` always runs the full `max_len` steps and never
            #     stops at <eos>, so hidden states produced AFTER the caption ends
            #     are meaningless padding-ish noise. Averaging all 20 steps dilutes
            #     v_cap. Here we mask every step from the FIRST <eos> onward (keep the
            #     <eos> step itself) and average only the valid caption steps, so
            #     v_cap reflects the real caption content and is a stronger signal.
            # FA: `_decode_caption` hamishe hame `max_len` step ro ejra mikone va sar-e
            #     <eos> nemi-iste, pas hidden-state-haye baad az payan-e caption noise
            #     hastan. Miangin gereftan az har 20 step v_cap ro raqiq mikone. Inja
            #     az avvalin <eos> be baad (khod-e <eos> ro negah midarim) mask mizanim
            #     va faghat az step-haye motabar miangin migirim ta v_cap mohtava-ye
            #     vaghei-e caption ro neshun bede (signal-e ghavi-tar).
            hidden = torch.stack(hidden_steps, dim=1)  # (N, T, hidden_dim)
            # cap[:, 1:] are the generated tokens aligned 1-to-1 with hidden steps.
            gen = cap[:, 1:]  # (N, T)
            eos = (gen == 2)
            # number of <eos> strictly before position t; keep positions with 0.
            prev_eos = eos.long().cumsum(dim=1) - eos.long()
            mask = (prev_eos == 0).to(hidden.dtype).unsqueeze(-1)  # (N, T, 1)
            v_cap = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            return v_cap, cap.detach()

        with torch.no_grad():
            cap = self.generate_caption(image, question_ids)
        return self.encode_caption(cap), cap
