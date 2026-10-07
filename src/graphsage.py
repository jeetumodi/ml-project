import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGE(nn.Module):
    """
    4-layer GraphSAGE architecture with mean aggregation per paper specification.
    """
    def __init__(self, in_dim, hidden_dim, depth=4, dropout=0.2):
        super(GraphSAGE, self).__init__()
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList()
        for i in range(depth):
            din = in_dim if i == 0 else hidden_dim
            self.convs.append(SAGEConv(din, hidden_dim, aggr='mean'))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.depth - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x
