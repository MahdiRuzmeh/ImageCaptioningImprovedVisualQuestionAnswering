"""VQ stack for caption-augmented visual question answering.

Overview
--------
This module implements the **answering model** in *Image captioning improved visual question
answering*: combine global image features (ResNet-101), region proposals (Faster R-CNN FPN),
question encoding (GRU), relational refinement over regions, **fusion with a frozen caption
module**, and a dual-LSTM answer decoder.

Paper / thesis reference
-------------------------
Use your PDF *Image captioning improved visual question answering.pdf* alongside this file:

- **Stage ordering (two-phase training)** — Typically described in *Method* / *Training procedure*:
  first train the captioner on MSCOCO (`ImageCaptioner`), then freeze it inside ``VQAModel``.
- **Caption integration** — Align with the subsection that explains using **generated caption
  embeddings** (or caption-side semantics) as auxiliary signal for VQA; here that signal is
  ``v_cap`` from ``captioner.get_caption_embedding(images, q_ids)``, fused with attended regions
  ``v_att`` via ``fuse_mode`` (``"mul"`` or ``"add"``).
- **Visual attention** — Match the paper’s discussion of question-conditioned pooling over
  regions; implemented in ``_attend`` after ``RelationGNN``.

Tensor flow (high level)
------------------------
``images`` → ResNet global ``g`` + ROI regions ``local`` → ``RelationGNN`` → ``v_att``;
``q_ids`` → GRU → ``q``; captioner → ``v_cap``; fuse → ``v``; dual LSTM → answer logits.

Examples
--------
Training step with teacher forcing (pseudo-code)::

    # logits: (batch, answer_len-1, a_vocab_size)
    logits = model(images, q_ids, a_ids=answer_token_ids)
    loss = criterion(logits.reshape(-1, logits.size(-1)), answer_token_ids[:, 1:].reshape(-1))

Inference (greedy answer tokens)::

    logits = model(images, q_ids, a_ids=None, max_answer_len=6)
    pred_ids = logits.argmax(dim=-1)
"""

from typing import Optional

import torch
from torch import nn
from torchvision.models import ResNet101_Weights, resnet101
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn

from .base_captioner import BaseImageCaptioner


class RelationGNN(nn.Module):
    """Lightweight pairwise message passing over region embeddings.

    For each region :math:`i`, aggregates edge messages from all pairs :math:`(i,j)`, then
    updates node features. This mirrors *relational reasoning between detected objects* often
    discussed in VQA architectures; cite the corresponding subsection in the thesis where
    region interaction / graph structure is defined.

    Args:
        dim: Hidden size per region (matches ``VQAModel.hidden_dim``).

    Examples:
        >>> import torch
        >>> gnn = RelationGNN(dim=512)
        >>> x = torch.randn(2, 32, 512)  # batch=2, regions=32
        >>> y = gnn(x)
        >>> y.shape
        torch.Size([2, 32, 512])
    """

    def __init__(self, dim: int = 512) -> None:
        super().__init__()
        self.edge = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.node = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one relational update step.

        Args:
            x: Region tensor of shape ``(batch, num_regions, dim)``.

        Returns:
            Updated regions of the same shape.
        """
        b, k, d = x.shape
        xi = x.unsqueeze(2).expand(b, k, k, d)
        xj = x.unsqueeze(1).expand(b, k, k, d)
        e = self.edge(torch.cat([xi, xj], dim=-1)).mean(dim=2)
        return self.node(torch.cat([x, e], dim=-1))


class VQAModel(nn.Module):
    """End-to-end VQA with caption-augmented fusion and dual-LSTM answer decoding.

    Paper mapping (adjust section labels to your PDF):

    - **Question encoder** — ``q_emb`` + ``q_gru`` + ``q_proj``: aligns with the paper’s
      sequential encoding of the question before attending visuals.
    - **Visual streams** — Global ResNet + Faster R-CNN ROI stream: cite *visual feature
      extraction* / *object regions*.
    - **Caption pathway** — Frozen ``captioner``: cite *image captioning module* and its use as
      **fixed** auxiliary evidence during VQA training.
    - **Fusion** — ``v = v_cap * v_att`` or ``v_cap + v_att``: cite *multimodal fusion* /
      *caption–visual combination*.
    - **Answer decoder** — ``lstm_att`` / ``lstm_ans``: cite *answer generation* LSTM stack.

    Args:
        q_vocab_size: Size of question vocabulary (includes specials).
        a_vocab_size: Size of answer vocabulary.
        pad_id: Padding index shared by question embeddings.
        captioner: Frozen ``BaseImageCaptioner`` (loaded from ``ImageCaptioner`` checkpoint).
        word_dim: Question / answer token embedding dimension.
        hidden_dim: Core hidden size for fusion and LSTMs.
        question_dim: GRU hidden size before ``q_proj``.
        max_regions: ROI features kept per image after truncation/padding.
        fuse_mode: ``"mul"`` element-wise product or ``"add"`` sum for ``v_cap`` and ``v_att``.

    Examples:
        Constructing the model (after loading captioner)::

            captioner = load_captioner(cfg, len(qv.itos), qv.pad_id, device)
            model = VQAModel(
                len(qv.itos), len(av.itos), qv.pad_id, captioner,
                cfg["word_dim"], cfg["hidden_dim"], cfg["question_dim"],
                cfg["max_regions"], cfg["fuse_mode"],
            ).to(device)
    """

    def __init__(
        self,
        q_vocab_size: int,
        a_vocab_size: int,
        pad_id: int,
        captioner: BaseImageCaptioner,
        word_dim: int = 512,
        hidden_dim: int = 512,
        question_dim: int = 1280,
        max_regions: int = 32,
        fuse_mode: str = "mul",
    ) -> None:
        super().__init__()
        self.captioner = captioner
        self.max_regions = max_regions
        self.hidden_dim = hidden_dim
        self.fuse_mode = fuse_mode

        rn = resnet101(weights=ResNet101_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(rn.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.g_proj = nn.Linear(2048, hidden_dim)

        det = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        self.detector = det
        for p in self.detector.parameters():
            p.requires_grad = False
        self.local_proj = nn.Linear(1024, hidden_dim)

        self.q_emb = nn.Embedding(q_vocab_size, word_dim, padding_idx=pad_id)
        self.q_gru = nn.GRU(word_dim, question_dim, batch_first=True)
        self.q_proj = nn.Linear(question_dim, hidden_dim)

        self.gnn = RelationGNN(hidden_dim)
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attn_score = nn.Linear(hidden_dim, 1)

        self.a_emb = nn.Embedding(a_vocab_size, word_dim, padding_idx=pad_id)
        self.lstm_att = nn.LSTMCell(word_dim + hidden_dim + hidden_dim, hidden_dim)
        self.lstm_ans = nn.LSTMCell(hidden_dim + hidden_dim + hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, a_vocab_size)

    @torch.no_grad()
    def _regions(self, images: torch.Tensor) -> torch.Tensor:
        """Extract fixed-count ROI features per image from a frozen Faster R-CNN FPN backbone.

        Paper reference: tie this to *object-level* or *region-based* visual representations.

        Args:
            images: Batch of tensors ``(N, 3, H, W)`` (typically 448×448).

        Returns:
            Tensor ``(N, max_regions, hidden_dim)`` after linear projection and padding.
        """
        t, _ = self.detector.transform(list(images), None)
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
                c = torch.cat([c, torch.zeros((self.max_regions - c.size(0), c.size(1)), device=c.device)], dim=0)
            out.append(c)
        return self.local_proj(torch.stack(out, dim=0))

    def _attend(self, regions: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Question-guided softmax attention over region embeddings.

        Args:
            regions: ``(batch, num_regions, dim)`` after relational encoding.
            q: Global question vector ``(batch, dim)``.

        Returns:
            Weighted sum ``(batch, dim)`` — attended visual context ``v_att``.
        """
        b, k, d = regions.shape
        q = q.unsqueeze(1).expand(b, k, d)
        s = torch.tanh(self.attn(torch.cat([regions, q], dim=-1)))
        a = torch.softmax(self.attn_score(s).squeeze(-1), dim=-1)
        return torch.einsum("bk,bkd->bd", a, regions)

    def forward(self, images: torch.Tensor, q_ids: torch.Tensor, a_ids: Optional[torch.Tensor] = None, max_answer_len: int = 6) -> torch.Tensor:
        """Run the full VQA forward pass.

        **Paper alignment:** Caption embeddings must use the **same** ``q_ids`` as the rest of the
        model so inference matches the thesis description of *question-conditioned* caption context.
        Train VQA: ``differentiable=True`` ta ``q_emb`` captioner az answer loss update beshe
        (bedoon caption GT — grad az LSTM hidden pool).

        Args:
            images: ``(batch, 3, H, W)``.
            q_ids: Padded question token ids ``(batch, seq_q)``.
            a_ids: If provided, gold answer ids for teacher forcing ``(batch, seq_a)`` (includes
                BOS/EOS framing consistent with the dataset). If ``None``, runs greedy decoding
                for ``max_answer_len - 1`` steps using predicted tokens.
            max_answer_len: Maximum generated answer length when ``a_ids`` is ``None``.

        Returns:
            Answer logits ``(batch, time, a_vocab_size)`` where ``time`` is ``seq_a - 1`` or
            ``max_answer_len - 1``.

        Examples:
            Teacher forcing (training)::

                logits = model(images, q_ids, a_ids=answers)

            Greedy evaluation::

                logits = model(images, q_ids, a_ids=None, max_answer_len=cfg["max_answer_len"])
                pred = logits.argmax(dim=-1)
        """
        g = self.g_proj(self.pool(self.resnet(images)).flatten(1))
        local = self._regions(images)

        _, h = self.q_gru(self.q_emb(q_ids))
        q = self.q_proj(h[-1])

        rel = self.gnn(local)
        v_att = self._attend(rel, q)

        # v_cap: train → LSTM hidden pool + grad be captioner.q_emb; eval → word_emb(caption)
        v_cap, _ = self.captioner.get_caption_embedding(
            images, q_ids, differentiable=self.training
        )

        v = v_cap * v_att if self.fuse_mode == "mul" else v_cap + v_att

        b = images.size(0)
        h1 = torch.zeros((b, self.hidden_dim), device=images.device)
        c1 = torch.zeros_like(h1)
        h2 = torch.zeros_like(h1)
        c2 = torch.zeros_like(h1)

        if a_ids is None:
            steps = max_answer_len - 1
            prev = torch.full((b,), 1, dtype=torch.long, device=images.device)
        else:
            steps = a_ids.size(1) - 1
            prev = a_ids[:, 0]

        logits = []
        for t in range(steps):
            a_prev = self.a_emb(prev)
            h1, c1 = self.lstm_att(torch.cat([a_prev, g, h2], dim=-1), (h1, c1))
            h2, c2 = self.lstm_ans(torch.cat([h1, h2, v], dim=-1), (h2, c2))
            logit = self.out(h2)
            logits.append(logit)
            prev = logit.argmax(dim=-1) if a_ids is None else a_ids[:, t + 1]

        return torch.stack(logits, dim=1)
