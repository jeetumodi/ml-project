import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedMultiHeadAttention(nn.Module):
    """
    Faithful implementation of EGCPPIS's Gated Multi-Head Self-Attention mechanism.
    Supports variable hidden_size and num_attention_heads (e.g. 10 heads).
    """
    def __init__(self, hidden_size, num_attention_heads=10):
        super(GatedMultiHeadAttention, self).__init__()
        assert hidden_size % num_attention_heads == 0, f"hidden_size {hidden_size} must be divisible by num_attention_heads {num_attention_heads}"
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.sigmoid = nn.Sigmoid()

    def transpose_for_scores(self, x):
        # x: (B, N, hidden_size)
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.head_dim)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)  # (B, heads, N, head_dim)

    def forward(self, x):
        """
        x: (B, N, hidden_size) where B is typically 1
        """
        norm_x = self.layer_norm(x)
        q = self.query(norm_x)
        k = self.key(norm_x)
        v = self.value(norm_x)
        gate = self.sigmoid(self.gate(norm_x))

        q_heads = self.transpose_for_scores(q)
        k_heads = self.transpose_for_scores(k)
        v_heads = self.transpose_for_scores(v)

        # Scaled dot-product: (B, heads, N, N)
        scores = torch.matmul(q_heads, k_heads.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)

        # Context: (B, heads, N, head_dim)
        context = torch.matmul(attn_weights, v_heads)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(x.size(0), x.size(1), self.hidden_size)

        # Apply gating
        gated_output = gate * context
        # Residual connection
        output = x + gated_output
        return output, attn_weights
