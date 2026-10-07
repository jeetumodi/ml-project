import torch
import numpy as np

def build_residue_graph(distance_matrix, cutoff=16.0):
    """
    Constructs edge_index and adjacency mask from distance matrix.
    Rule from paper and official code:
    edge exists if dist < cutoff or if i == j (self-loop).
    """
    L = distance_matrix.shape[0]
    # Boolean adjacency matrix
    adj = (distance_matrix < cutoff) | (np.eye(L, dtype=bool))
    
    # Nonzero rows and cols for sparse representation
    rows, cols = np.nonzero(adj)
    edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.int64)
    
    # Distance values along edges
    edge_dists = torch.tensor(distance_matrix[rows, cols], dtype=torch.float32)
    
    return edge_index, edge_dists, torch.tensor(adj, dtype=torch.float32)
