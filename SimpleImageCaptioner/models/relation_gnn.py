"""RelationGNN — message passing roye region-ha (paper §3.1, §3.3)."""

from torch import nn
import torch


class RelationGNN(nn.Module):
    """Finglish — RelationGNN (paper §3.1):
        Har region = yek node; har joft (i,j) = yek edge.
        Mesal: person+bicycle → edge message «riding»; dog+ball → «chasing».
        Khoroji: feature har region ba context hamsaye hash update mishavad.
    """

    def __init__(self, dim: int = 512) -> None:
        super().__init__()
        self.edge = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, dim)
        )
        self.node = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, num_regions, dim) → updated region features, same shape.

        Finglish — Bug 4 fix: residual connection ezafe shod.
        Ghabl: khoroji GNN be kolli region feature haye asli ro replace mikard.
        Hala: delta (taghir) be x asli ezafe mishe → gradient path hifz mishe
        va age GNN chizi yad nagirad, x asli barmigarde (training stable mishe).
        """
        b, k, d = x.shape
        xi = x.unsqueeze(2).expand(b, k, k, d)
        xj = x.unsqueeze(1).expand(b, k, k, d)
        edge_msg = self.edge(torch.cat([xi, xj], dim=-1)).mean(dim=2)
        return x + self.node(torch.cat([x, edge_msg], dim=-1))
