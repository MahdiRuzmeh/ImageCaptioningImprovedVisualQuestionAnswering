from typing import Optional

import torch
from torch import nn
from torchvision.models import ResNet101_Weights, resnet101
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn

from .base_captioner import BaseImageCaptioner


class RelationGNN(nn.Module):
    def __init__(self, dim: int = 512) -> None:
        super().__init__()
        self.edge = nn.Sequential(nn.Linear(dim*2, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.node = nn.Sequential(nn.Linear(dim*2, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b,k,d = x.shape
        xi = x.unsqueeze(2).expand(b,k,k,d)
        xj = x.unsqueeze(1).expand(b,k,k,d)
        e = self.edge(torch.cat([xi,xj], dim=-1)).mean(dim=2)
        return self.node(torch.cat([x,e], dim=-1))


class VQAModel(nn.Module):
    def __init__(self, q_vocab_size: int, a_vocab_size: int, pad_id: int, captioner: BaseImageCaptioner, word_dim: int = 512, hidden_dim: int = 512, question_dim: int = 1280, max_regions: int = 32, fuse_mode: str = "mul") -> None:
        super().__init__()
        self.captioner = captioner
        self.max_regions = max_regions
        self.hidden_dim = hidden_dim
        self.fuse_mode = fuse_mode

        rn = resnet101(weights=ResNet101_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(rn.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1,1))
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
        self.attn = nn.Linear(hidden_dim*2, hidden_dim)
        self.attn_score = nn.Linear(hidden_dim, 1)

        self.a_emb = nn.Embedding(a_vocab_size, word_dim, padding_idx=pad_id)
        self.lstm_att = nn.LSTMCell(word_dim + hidden_dim + hidden_dim, hidden_dim)
        self.lstm_ans = nn.LSTMCell(hidden_dim + hidden_dim + hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, a_vocab_size)

    @torch.no_grad()
    def _regions(self, images: torch.Tensor) -> torch.Tensor:
        t,_ = self.detector.transform(list(images), None)
        feats = self.detector.backbone(t.tensors)
        props,_ = self.detector.rpn(t, feats, None)
        roi = self.detector.roi_heads.box_roi_pool(feats, props, t.image_sizes)
        roi = self.detector.roi_heads.box_head(roi)
        counts = [len(p) for p in props]
        chunks = torch.split(roi, counts)
        out = []
        for c in chunks:
            c = c[: self.max_regions]
            if c.size(0) < self.max_regions:
                c = torch.cat([c, torch.zeros((self.max_regions-c.size(0), c.size(1)), device=c.device)], dim=0)
            out.append(c)
        return self.local_proj(torch.stack(out, dim=0))

    def _attend(self, regions: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        b,k,d = regions.shape
        q = q.unsqueeze(1).expand(b,k,d)
        s = torch.tanh(self.attn(torch.cat([regions,q], dim=-1)))
        a = torch.softmax(self.attn_score(s).squeeze(-1), dim=-1)
        return torch.einsum("bk,bkd->bd", a, regions)

    def forward(self, images: torch.Tensor, q_ids: torch.Tensor, a_ids: Optional[torch.Tensor] = None, max_answer_len: int = 6) -> torch.Tensor:
        g = self.g_proj(self.pool(self.resnet(images)).flatten(1))
        local = self._regions(images)

        _,h = self.q_gru(self.q_emb(q_ids))
        q = self.q_proj(h[-1])

        rel = self.gnn(local)
        v_att = self._attend(rel, q)

        with torch.no_grad():
            v_cap,_ = self.captioner.get_caption_embedding(images, q_ids)

        v = v_cap * v_att if self.fuse_mode == "mul" else v_cap + v_att

        b = images.size(0)
        h1 = torch.zeros((b,self.hidden_dim), device=images.device)
        c1 = torch.zeros_like(h1)
        h2 = torch.zeros_like(h1)
        c2 = torch.zeros_like(h1)

        if a_ids is None:
            steps = max_answer_len - 1
            prev = torch.full((b,), 1, dtype=torch.long, device=images.device)
        else:
            steps = a_ids.size(1)-1
            prev = a_ids[:,0]

        logits = []
        for t in range(steps):
            a_prev = self.a_emb(prev)
            h1,c1 = self.lstm_att(torch.cat([a_prev, g, h2], dim=-1), (h1,c1))
            h2,c2 = self.lstm_ans(torch.cat([h1, h2, v], dim=-1), (h2,c2))
            logit = self.out(h2)
            logits.append(logit)
            prev = logit.argmax(dim=-1) if a_ids is None else a_ids[:,t+1]

        return torch.stack(logits, dim=1)
