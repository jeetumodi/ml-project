import torch
import torch.nn as nn
import torch.nn.functional as F

from .egnn import EGNN
from .graphsage import GraphSAGE
from .contrastive import GraphContrastiveLearning, construct_meta_pos_mask
from .attention import GatedMultiHeadAttention

class BinaryClassificationMLP(nn.Module):
    def __init__(self, in_dim, hidden1=128, hidden2=16, dropout=0.2):
        super(BinaryClassificationMLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        logits = self.out(x)
        probs = torch.sigmoid(logits)
        return probs

class EGCPPISModel(nn.Module):
    """
    Faithful EGCPPIS Architecture Reproduction:
    - Residue branch: 4-layer EGNN
    - Topology branch: 4-layer GraphSAGE
    - Contrastive Learning: Dual InfoNCE loss on top 10% positive pairs
    - Fusion: 2 * feature_dim representation
    - Attention: 10-head Gated Multi-Head Self-Attention
    - Classifier: Two-layer MLP with Sigmoid output
    """
    def __init__(self, feature_dim=74, num_heads=10, dropout=0.2, cl_temperature=0.8, cl_lambda=0.5):
        super(EGCPPISModel, self).__init__()
        self.feature_dim = feature_dim
        
        # Round feature_dim up if needed so it is divisible by num_heads
        if feature_dim % num_heads != 0:
            self.aligned_dim = ((feature_dim // num_heads) + 1) * num_heads
            self.input_align = nn.Linear(feature_dim, self.aligned_dim)
        else:
            self.aligned_dim = feature_dim
            self.input_align = nn.Identity()

        # 1. EGNN Residue geometric branch
        self.egnn = EGNN(self.aligned_dim, hidden_dim=self.aligned_dim, depth=4)

        # 2. GraphSAGE topology branch
        self.sage = GraphSAGE(self.aligned_dim, self.aligned_dim, depth=4, dropout=dropout)

        # 3. Contrastive Learning Module
        self.cl = GraphContrastiveLearning(self.aligned_dim, temperature=cl_temperature, lambda_1=cl_lambda)

        # 4. Gated Multi-Head Attention on fused representation (dim = 2 * aligned_dim)
        self.fused_dim = self.aligned_dim * 2
        self.attention = GatedMultiHeadAttention(self.fused_dim, num_attention_heads=num_heads)

        # 5. Classifier MLP
        self.classifier = BinaryClassificationMLP(self.fused_dim, hidden1=128, hidden2=16, dropout=dropout)

    def forward(self, node_feats, coords, edge_index, dist_matrix):
        """
        node_feats: (N, feature_dim)
        coords: (N, 3)
        edge_index: (2, E)
        dist_matrix: (N, N)
        """
        N = node_feats.size(0)
        h = self.input_align(node_feats)

        # Branch 1: EGNN
        h_egnn, _ = self.egnn(h, coords, edge_index)

        # Branch 2: GraphSAGE
        h_sage = self.sage(h, edge_index)

        # Contrastive Learning & Fusion
        pos_mask = construct_meta_pos_mask(N, dist_matrix, top_k_ratio=0.1)
        cl_loss, fused = self.cl(h_egnn, h_sage, pos_mask)

        # Gated Multi-Head Attention
        fused_seq = fused.unsqueeze(0)  # (1, N, fused_dim)
        attn_out, _ = self.attention(fused_seq)
        attn_out = attn_out.squeeze(0)  # (N, fused_dim)

        # Final Classification
        probs = self.classifier(attn_out)  # (N, 1)

        return probs, cl_loss
