import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def construct_meta_pos_mask(node_number, adj_dist, top_k_ratio=0.1):
    """
    Selects top 10% closest residues as positive sample pairs.
    adj_dist: (N, N) distance or similarity matrix
    """
    device = adj_dist.device
    dist_np = adj_dist.detach().cpu().numpy()
    target_pos = np.zeros((node_number, node_number), dtype=np.float32)
    num_pos = max(1, int(node_number * top_k_ratio))

    for i in range(node_number):
        target_pos[i, i] = 1.0  # Self is positive
        row = dist_np[i]
        # Sort ascending for distances (closest residues)
        closest_indices = np.argsort(row)[:num_pos]
        target_pos[i, closest_indices] = 1.0

    return torch.tensor(target_pos, device=device)

class GraphContrastiveLearning(nn.Module):
    """
    EGCPPIS Contrastive Learning Module:
    Two-layer MLP projection, Cosine similarity, Top-10% positive selection,
    Dual InfoNCE loss across views.
    """
    def __init__(self, hidden_dim, temperature=0.8, lambda_1=0.5):
        super(GraphContrastiveLearning, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.temperature = temperature
        self.lambda_1 = lambda_1

        for fc in self.project:
            if isinstance(fc, nn.Linear):
                nn.init.xavier_normal_(fc.weight, gain=1.414)

    def compute_similarity(self, view1, view2):
        v1_norm = torch.norm(view1, dim=-1, keepdim=True).clamp(min=1e-8)
        v2_norm = torch.norm(view2, dim=-1, keepdim=True).clamp(min=1e-8)
        dot_product = torch.mm(view1, view2.t())
        norm_product = torch.mm(v1_norm, v2_norm.t()).clamp(min=1e-8)
        sim_matrix = torch.exp(dot_product / (norm_product * self.temperature))
        return sim_matrix

    def forward(self, view1, view2, pos_mask):
        """
        view1: (N, hidden_dim)
        view2: (N, hidden_dim)
        pos_mask: (N, N) binary mask
        """
        proj1 = self.project(view1)
        proj2 = self.project(view2)

        sim_12 = self.compute_similarity(proj1, proj2)
        sim_21 = sim_12.t()

        # Row-normalized probabilities
        prob_12 = sim_12 / (torch.sum(sim_12, dim=1, keepdim=True) + 1e-8)
        prob_21 = sim_21 / (torch.sum(sim_21, dim=1, keepdim=True) + 1e-8)

        # InfoNCE loss
        loss_1 = -torch.log((prob_12 * pos_mask).sum(dim=-1).clamp(min=1e-8)).mean()
        loss_2 = -torch.log((prob_21 * pos_mask).sum(dim=-1).clamp(min=1e-8)).mean()

        cl_loss = self.lambda_1 * loss_1 + (1.0 - self.lambda_1) * loss_2
        fused_embedding = torch.cat([proj1, proj2], dim=-1)

        return cl_loss, fused_embedding
