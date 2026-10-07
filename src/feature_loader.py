import os
import numpy as np
import torch

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

def sequence_to_one_hot(sequence):
    """Encodes amino acid sequence as N x 20 one-hot numpy array."""
    L = len(sequence)
    one_hot = np.zeros((L, 20), dtype=np.float32)
    for i, aa in enumerate(sequence):
        if aa in AMINO_ACIDS:
            one_hot[i, AMINO_ACIDS.index(aa)] = 1.0
    return one_hot

class FeatureLoader:
    def __init__(self, data_root="data"):
        self.data_root = data_root
        self.feature_base = os.path.join(data_root, "GraphPPIS", "Feature")
        self.dssp_dir = os.path.join(self.feature_base, "dssp")
        self.pssm_dir = os.path.join(self.feature_base, "pssm")
        self.hmm_dir = os.path.join(self.feature_base, "hmm")
        self.dist_dir = os.path.join(self.feature_base, "distance_map")

    def load_protein_features(self, pid, sequence):
        """
        Loads all available features for a given protein ID:
        - one_hot: (L, 20)
        - dssp: (L, 14)
        - pssm: (L, 20)
        - hmm: (L, 20)
        - distance_map: (L, L)
        Returns:
        residue_features: (L, 74) float32
        distance_map: (L, L) float32
        """
        L = len(sequence)
        one_hot = sequence_to_one_hot(sequence)
        
        # Load from disk
        dssp_path = os.path.join(self.dssp_dir, f"{pid}.npy")
        pssm_path = os.path.join(self.pssm_dir, f"{pid}.npy")
        hmm_path = os.path.join(self.hmm_dir, f"{pid}.npy")
        dist_path = os.path.join(self.dist_dir, f"{pid}.npy")
        
        if not (os.path.exists(dssp_path) and os.path.exists(pssm_path) and os.path.exists(hmm_path) and os.path.exists(dist_path)):
            return None, None
            
        dssp = np.load(dssp_path).astype(np.float32)
        pssm = np.load(pssm_path).astype(np.float32)
        hmm = np.load(hmm_path).astype(np.float32)
        dist = np.load(dist_path).astype(np.float32)
        
        # Validate lengths
        if not (len(dssp) == L and len(pssm) == L and len(hmm) == L and dist.shape[0] == L):
            return None, None
            
        # Concatenate residue features: 20 + 14 + 20 + 20 = 74 dims
        node_features = np.concatenate([one_hot, dssp, pssm, hmm], axis=1).astype(np.float32)
        
        return node_features, dist
