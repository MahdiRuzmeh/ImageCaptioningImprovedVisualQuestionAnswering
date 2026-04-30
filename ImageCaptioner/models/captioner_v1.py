from typing import Optional, Tuple

import torch
from torch import nn
from torchvision.models import ResNet101_Weights, resnet101
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn

from .base_captioner import BaseImageCaptioner


class ImageCaptionerV1(BaseImageCaptioner):
    def __init__(self, vocab_size: int, pad_id: int, word_dim: int = 512, hidden_dim: int = 512, max_regions: int = 32, question_dim: int = 1280) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_regions = max_regions
        self.emb = nn.Embedding(vocab_size, word_dim, padding_idx=pad_id)

        rn = resnet101(weights=ResNet101_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(rn.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_proj = nn.Linear(2048, hidden_dim)

        det = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        self.detector = det
        for p in self.detector.parameters():
            p.requires_grad = False
        self.detector.eval()
        self.local_proj = nn.Linear(1024, hidden_dim)

        self.q_proj = nn.Linear(question_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attn_score = nn.Linear(hidden_dim, 1)

        self.lstm = nn.LSTMCell(word_dim + hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def train(self, mode: bool = True) -> "ImageCaptionerV1":
        """Keep detector frozen in eval mode while training other modules."""
        super().train(mode)
        self.detector.eval()
        return self

    @torch.no_grad()
    def _regions(self, images: torch.Tensor) -> torch.Tensor:
        img_list = list(images)
        t,_ = self.detector.transform(img_list, None)
        feats = self.detector.backbone(t.tensors)
        props,_ = self.detector.rpn(t, feats, None)
        roi = self.detector.roi_heads.box_roi_pool(feats, props, t.image_sizes)
        roi = self.detector.roi_heads.box_head(roi)
        counts = [len(p) for p in props]
        chunks = torch.split(roi, counts)
        out = []
        for c in chunks:
            c = c[:self.max_regions]
            if c.size(0) < self.max_regions:
                pad = torch.zeros((self.max_regions - c.size(0), c.size(1)), device=c.device)
                c = torch.cat([c, pad], dim=0)
            out.append(c)
        return self.local_proj(torch.stack(out, dim=0))

    def _qctx(self, q: Optional[torch.Tensor], b: int, device: torch.device) -> torch.Tensor:
        if q is None:
            return torch.zeros((b, self.hidden_dim), device=device)
        qe = self.emb(q).mean(dim=1)
        if qe.size(-1) != self.hidden_dim:
            qe = self.q_proj(qe)
        return qe

    def _attend(self, local: torch.Tensor, qctx: torch.Tensor) -> torch.Tensor:
        b,k,d = local.shape
        q = qctx.unsqueeze(1).expand(b,k,d)
        s = torch.tanh(self.attn(torch.cat([local, q], dim=-1)))
        a = torch.softmax(self.attn_score(s).squeeze(-1), dim=-1)
        return torch.einsum("bk,bkd->bd", a, local)

    def _visual(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.resnet(images)
        g = self.global_proj(self.pool(f).flatten(1))
        l = self._regions(images)
        return g,l

    def forward_train(self, images: torch.Tensor, caption_ids: torch.Tensor, question_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        _,local = self._visual(images)
        ctx = self._attend(local, self._qctx(question_ids, images.size(0), images.device))
        h = torch.zeros((images.size(0), self.hidden_dim), device=images.device)
        c = torch.zeros_like(h)
        logits = []
        for t in range(caption_ids.size(1)-1):
            w = self.emb(caption_ids[:,t])
            h,c = self.lstm(torch.cat([w,ctx], dim=-1), (h,c))
            logits.append(self.out(h))
        return torch.stack(logits, dim=1)

    @torch.no_grad()
    def generate_caption(self, image: torch.Tensor, question_ids: Optional[torch.Tensor] = None, max_len: int = 20) -> torch.Tensor:
        _,local = self._visual(image)
        ctx = self._attend(local, self._qctx(question_ids, image.size(0), image.device))
        h = torch.zeros((image.size(0), self.hidden_dim), device=image.device)
        c = torch.zeros_like(h)
        tok = torch.full((image.size(0),), 1, dtype=torch.long, device=image.device)
        out = [tok]
        for _ in range(max_len-1):
            w = self.emb(tok)
            h,c = self.lstm(torch.cat([w,ctx], dim=-1), (h,c))
            tok = self.out(h).argmax(dim=-1)
            out.append(tok)
        return torch.stack(out, dim=1)

    def encode_caption(self, caption_ids: torch.Tensor) -> torch.Tensor:
        return self.emb(caption_ids).mean(dim=1)

    @torch.no_grad()
    def get_caption_embedding(self, image: torch.Tensor, question_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        cap = self.generate_caption(image, question_ids)
        return self.encode_caption(cap), cap
