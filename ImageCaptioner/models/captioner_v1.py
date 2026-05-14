"""Concrete captioning submodule — feeds frozen embeddings into VQA (*thesis paper*).

Problem addressed (see *Image captioning improved visual question answering.pdf*)
-------------------------------------------------------------------------------
Natural-language questions often benefit from **dense scene descriptions**. This implementation
learns to describe MSCOCO images; during VQA training its greedy caption is pooled into a
vector ``encode_caption(generate_caption(...))`` consumed inside ``VQAModel.forward``.

Architecture mapping (cite matching figures/sections in your PDF)
-----------------------------------------------------------------
1. **CNN backbone** — ResNet-101 truncated before classification; global average pooling ->
   ``global_proj`` vector (scene gist).
2. **Region branch** — Faster R-CNN RPN + ROI head features (frozen detector weights), projected to
   ``hidden_dim``, truncated/padded to ``max_regions``.
3. **Question-guided attention** — Optional token ids embedded and mean-pooled into ``qctx``;
   attends over regions → fixed ``ctx`` fed to LSTM at every caption timestep (paper: tying
   linguistic query to salient objects).
4. **Caption LSTM** — ``LSTMCell`` consuming ``[word_emb ; ctx]``; trains with teacher forcing.

Examples
--------
Training batch::

    logits = model.forward_train(images, caption_ids)  # question_ids optional
    loss = CrossEntropyLoss(ignore_index=pad)(logits.flatten(0,1), caption_ids[:,1:].flatten())

Frozen embedding for VQA (already wrapped in ``get_caption_embedding``)::

    vec, ids = captioner.get_caption_embedding(images, question_ids=q_ids)

References
----------
Cross-reference thesis sections on **caption module**, **region attention**, and **embedding
extraction** used by downstream fusion (multiply/add with attended ROI features).
"""

from typing import Optional, Tuple

import torch
from torch import nn
from torchvision.models import ResNet101_Weights, resnet101
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn

from .base_captioner import BaseImageCaptioner


class ImageCaptionerV1(BaseImageCaptioner):
    """Region-attended LSTM captioner with frozen Faster R-CNN detector.

    Args:
        vocab_size: Caption vocabulary size after specials (must match checkpoint row count when
            reloading identical vocab).
        pad_id: Padding token index for ``nn.Embedding``.
        word_dim: Token embedding dimension.
        hidden_dim: Region/context/LSTM hidden size (must align with ``VQAModel.hidden_dim`` when
            sharing dims for fusion).
        max_regions: Number of ROI vectors retained per image.
        question_dim: Width fed into ``q_proj`` when ``_qctx`` projects pooled questions to
            ``hidden_dim`` (use ``word_dim`` if ``word_dim != hidden_dim``; ignored when widths match).

    Examples:
        Instantiate like ``training/train.py``::

            m = ImageCaptionerV1(len(vocab.itos), vocab.pad_id, word_dim=512,
                                 hidden_dim=512, max_regions=32, question_dim=1280)
    """

    def __init__(
            self,
            vocab_size: int,
            pad_id: int,
            word_dim: int = 512,
            hidden_dim: int = 512,
            max_regions: int = 32,
            question_dim: int = 1280
    ) -> None:
        """Register submodules and freeze the detection backbone.

        **Data flow (batch ``N``)** — This method only *builds* layers; the live path is
        ``_visual`` → ``_qctx`` / ``_attend`` → ``LSTMCell`` in ``forward_train`` /
        ``generate_caption``.

        1. **``emb``** — Maps token ids ``(N, T)`` to ``(N, T, word_dim)``. Captions and optional
           ``question_ids`` share this table. ``padding_idx`` stops PAD from receiving gradients.

        2. **ResNet-101** — ``resnet101`` loads pretrained weights. ``list(rn.children())[:-2]``
           removes the last two modules (global pool + classifier), so you keep a **4D** feature
           map ``(N, 2048, h, w)`` instead of 1000 class scores. ``pool`` + ``global_proj`` produce
           a global vector ``(N, hidden_dim)``. Call sites use ``_, local = self._visual(...)``, so
           that global vector is **not** fed into the caption LSTM today; only ROI attention does.

        3. **Faster R-CNN FPN** — Full detector (backbone, RPN, ROI heads). ``_regions`` reads
           1024-D ROI features, then ``local_proj`` maps each to ``hidden_dim`` and pads/truncates
           to ``max_regions`` → ``(N, max_regions, hidden_dim)``. ``requires_grad = False`` and
           ``eval()`` freeze weights and BatchNorm stats while caption layers train.
           ``super().__init__()`` runs first so every ``nn.*`` is registered for ``.parameters()``.

        4. **Attention** — ``_qctx`` mean-pools question embeddings to ``(N, word_dim)``; if that
           width differs from ``hidden_dim``, ``q_proj`` maps ``(N, question_dim)`` →
           ``(N, hidden_dim)`` (set ``question_dim`` equal to the pooled width you actually pass in,
           typically ``word_dim``). ``attn`` / ``attn_score`` softmax over regions → ``ctx``
           ``(N, hidden_dim)`` reused at every caption step.

        5. **Decoder** — ``LSTMCell`` input is ``cat(word_emb, ctx)`` → length ``word_dim +
           hidden_dim``; ``out`` maps hidden state to logits ``(N, vocab_size)`` per step.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_regions = max_regions
        # ids (N, T) -> (N, T, word_dim); shared for caption tokens and optional question_ids.
        self.emb = nn.Embedding(vocab_size, word_dim, padding_idx=pad_id)

        rn = resnet101(weights=ResNet101_Weights.DEFAULT)
        # Keep conv trunk only: drop global pool + ImageNet classifier (children [-2:]).
        self.resnet = nn.Sequential(*list(rn.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_proj = nn.Linear(2048, hidden_dim)

        det = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        self.detector = det
        for p in self.detector.parameters():
            p.requires_grad = False
        self.detector.eval()
        # Faster R-CNN box head outputs 1024-D per ROI before projection to hidden_dim.
        self.local_proj = nn.Linear(1024, hidden_dim)

        self.q_proj = nn.Linear(question_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attn_score = nn.Linear(hidden_dim, 1)

        self.lstm = nn.LSTMCell(word_dim + hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def train(self, mode: bool = True) -> "ImageCaptionerV1":
        """Call ``nn.Module.train`` but **keep detector submodules in eval** (BN stats frozen).

        Examples:
            Always safe inside caption training loop::

                model.train()  # detector stays eval internally
        """
        super().train(mode)
        self.detector.eval()
        return self

    @torch.no_grad()
    def _regions(self, images: torch.Tensor) -> torch.Tensor:
        """ROI-aligned vectors projected to ``hidden_dim``.

        Returns:
            Tensor shaped ``(batch, max_regions, hidden_dim)``.

        Paper reference: parallel *object proposals* stream feeding caption decoder context.
        """
        img_list = list(images)
        t, _ = self.detector.transform(img_list, None)
        feats = self.detector.backbone(t.tensors)
        props, _ = self.detector.rpn(t, feats, None)
        roi = self.detector.roi_heads.box_roi_pool(feats, props, t.image_sizes)
        roi = self.detector.roi_heads.box_head(roi)
        counts = [len(p) for p in props]
        chunks = torch.split(roi, counts)
        out = []
        for c in chunks:
            c = c[: self.max_regions]
            if c.size(0) < self.max_regions:
                pad = torch.zeros(
                    (self.max_regions - c.size(0), c.size(1)), device=c.device)
                c = torch.cat([c, pad], dim=0)
            out.append(c)
        return self.local_proj(torch.stack(out, dim=0))

    def _qctx(self, q: Optional[torch.Tensor], b: int, device: torch.device) -> torch.Tensor:
        """Pool optional ``question_ids`` into one vector per batch row.

        If ``q is None`` (caption-only MSCOCO training), returns zeros so attention reduces to a
        learned bias distribution—still valid but **not** question-conditioned.

        When ``VQAModel`` passes ``q_ids``, cite thesis alignment with **question-aware captions**.
        """
        if q is None:
            return torch.zeros((b, self.hidden_dim), device=device)
        qe = self.emb(q).mean(dim=1)
        if qe.size(-1) != self.hidden_dim:
            qe = self.q_proj(qe)
        return qe

    def _attend(self, local: torch.Tensor, qctx: torch.Tensor) -> torch.Tensor:
        """Scaled softmax attention over ``local`` guided by ``qctx``.

        Args:
            local: ``(batch, max_regions, hidden_dim)``.
            qctx: ``(batch, hidden_dim)``.

        Returns:
            Context ``(batch, hidden_dim)`` concatenated channel-wise with word embeddings each step.
        """
        b, k, d = local.shape
        q = qctx.unsqueeze(1).expand(b, k, d)
        s = torch.tanh(self.attn(torch.cat([local, q], dim=-1)))
        a = torch.softmax(self.attn_score(s).squeeze(-1), dim=-1)
        return torch.einsum("bk,bkd->bd", a, local)

    def _visual(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute global ResNet map embedding and ROI stack.

        Returns:
            ``(global_hidden, region_hidden)`` both with trailing dim ``hidden_dim``.
        """
        f = self.resnet(images)
        g = self.global_proj(self.pool(f).flatten(1))
        l = self._regions(images)
        return g, l

    def forward_train(self, images: torch.Tensor, caption_ids: torch.Tensor, question_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Teacher-forced next-token prediction.

        Args:
            images: ``(N, 3, H, W)``.
            caption_ids: ``(N, T)`` token ids including BOS at ``[:,0]``.
            question_ids: Optional ``(N, Tq)`` padded question tokens for attention biasing.

        Returns:
            Logits ``(N, T-1, vocab_size)`` predicting tokens ``caption_ids[:,1:]``.

        Examples:
            Matching CE target::

                logits = model.forward_train(imgs, caps)
                loss = crit(logits.reshape(-1, V), caps[:, 1:].reshape(-1))
        """
        _, local = self._visual(images)
        ctx = self._attend(local, self._qctx(
            question_ids, images.size(0), images.device))
        h = torch.zeros((images.size(0), self.hidden_dim),
                        device=images.device)
        c = torch.zeros_like(h)
        logits = []
        for t in range(caption_ids.size(1) - 1):
            w = self.emb(caption_ids[:, t])
            h, c = self.lstm(torch.cat([w, ctx], dim=-1), (h, c))
            logits.append(self.out(h))
        return torch.stack(logits, dim=1)

    @torch.no_grad()
    def generate_caption(self, image: torch.Tensor, question_ids: Optional[torch.Tensor] = None, max_len: int = 20) -> torch.Tensor:
        """Greedy decoding starting from BOS token id ``1``.

        Args:
            image: ``(N, 3, H, W)``.
            question_ids: Same semantics as ``forward_train``.
            max_len: Maximum caption length including BOS.

        Returns:
            ``(N, max_len)`` token ids (EOS may appear before last column).

        Examples:
            Used internally by ``get_caption_embedding`` during VQA fusion.
        """
        _, local = self._visual(image)
        ctx = self._attend(local, self._qctx(
            question_ids, image.size(0), image.device))
        h = torch.zeros((image.size(0), self.hidden_dim), device=image.device)
        c = torch.zeros_like(h)
        tok = torch.full((image.size(0),), 1,
                         dtype=torch.long, device=image.device)
        out = [tok]
        for _ in range(max_len - 1):
            w = self.emb(tok)
            h, c = self.lstm(torch.cat([w, ctx], dim=-1), (h, c))
            tok = self.out(h).argmax(dim=-1)
            out.append(tok)
        return torch.stack(out, dim=1)

    def encode_caption(self, caption_ids: torch.Tensor) -> torch.Tensor:
        """Mean-pool token embeddings along time → sentence vector ``(N, word_dim)``.

        Thesis reference: this vector is the **caption-side signal** fused with ROI attention in VQA.
        """
        return self.emb(caption_ids).mean(dim=1)

    @torch.no_grad()
    def get_caption_embedding(self, image: torch.Tensor, question_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Greedy caption → pooled embedding.

        Returns:
            ``(sentence_embedding, caption_token_ids)`` as required by ``VQAModel``.

        Examples:
            Inside ``VQAModel.forward``::

                v_cap, _ = captioner.get_caption_embedding(images, q_ids)
        """
        cap = self.generate_caption(image, question_ids)
        return self.encode_caption(cap), cap
