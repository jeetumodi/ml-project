import torch
import torch.nn as nn
import torch.nn.functional as F

class CoorsNorm(nn.Module):
    def __init__(self, eps=1e-8, scale_init=1e-2):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coors):
        norm = coors.norm(dim=-1, keepdim=True)
        normed_coors = coors / norm.clamp(min=self.eps)
        return normed_coors * self.scale

class EGCL(nn.Module):
    """
    Equivariant Graph Convolutional Layer (EGCL).
    Updates node features and coordinate embeddings while preserving E(n) equivariance.
    """
    def __init__(self, in_dim, out_dim, hidden_dim=128, m_dim=16, coor_weights_clamp_value=2.0):
        super(EGCL, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.coor_weights_clamp_value = coor_weights_clamp_value
        self.coors_norm = CoorsNorm()

        # Edge MLP: (h_i, h_j, ||x_i - x_j||^2) -> m_ij
        edge_input_dim = in_dim * 2 + 1
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, m_dim),
            nn.SiLU()
        )

        # Coordinates MLP: m_ij -> scalar coordinate update weight
        self.coors_mlp = nn.Sequential(
            nn.Linear(m_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

        # Node MLP: (h_i, aggregated_m_i) -> h_i'
        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim + m_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.node_norm = nn.LayerNorm(out_dim)

        # Initialize weights with small std to prevent explosion
        for m in [self.edge_mlp, self.coors_mlp, self.node_mlp]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=1e-3)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, h, x, edge_index):
        """
        h: (N, in_dim) node features
        x: (N, 3) spatial coordinates
        edge_index: (2, E)
        """
        row, col = edge_index[0], edge_index[1]

        # Coordinate differences & squared distances
        rel_coors = x[row] - x[col]
        dist_sq = torch.sum(rel_coors ** 2, dim=-1, keepdim=True).clamp(max=100.0)

        # Edge features
        edge_feat = torch.cat([h[row], h[col], dist_sq], dim=-1)
        m_ij = self.edge_mlp(edge_feat)

        # Coordinate update with CoorsNorm and clamping
        coor_weights = self.coors_mlp(m_ij).clamp(min=-self.coor_weights_clamp_value, max=self.coor_weights_clamp_value)
        normed_rel_coors = self.coors_norm(rel_coors)
        trans = normed_rel_coors * coor_weights
        agg_trans = torch.zeros_like(x).scatter_add_(0, row.unsqueeze(-1).expand_as(trans), trans)
        x_new = x + agg_trans

        # Aggregate messages to nodes
        agg_m = torch.zeros(h.size(0), m_ij.size(-1), device=h.device).scatter_add_(
            0, row.unsqueeze(-1).expand_as(m_ij), m_ij
        )

        # Node update
        node_input = torch.cat([h, agg_m], dim=-1)
        h_new = self.node_norm(self.node_mlp(node_input))
        if self.in_dim == self.out_dim:
            h_new = h + h_new

        return h_new, x_new

class EGNN(nn.Module):
    """
    4-layer EGNN architecture per EGCPPIS specification.
    """
    def __init__(self, in_dim, hidden_dim=128, depth=4):
        super(EGNN, self).__init__()
        self.depth = depth
        self.layers = nn.ModuleList([
            EGCL(in_dim if i == 0 else hidden_dim, hidden_dim, hidden_dim=hidden_dim * 2)
            for i in range(depth)
        ])
        self.out_proj = nn.Linear(hidden_dim, in_dim) if in_dim != hidden_dim else nn.Identity()

    def forward(self, h, x, edge_index):
        for layer in self.layers:
            h, x = layer(h, x, edge_index)
        return self.out_proj(h), x
