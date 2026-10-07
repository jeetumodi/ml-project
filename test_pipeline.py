import os
import torch
import numpy as np

from src.utils import set_seed, get_device
from src.data_loader import parse_fasta
from src.feature_loader import FeatureLoader
from src.graph_builder import build_residue_graph
from src.model import EGCPPISModel
from src.losses import EGCPPISLoss

def main():
    print("================================================================================")
    print("                     PRE-TRAINING SYSTEM & PIPELINE VALIDATION                  ")
    print("================================================================================")
    
    # 1. Environment Test
    set_seed(100000)
    device = get_device()
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    # 2. Dataset and Feature Loading Test
    loader = FeatureLoader(data_root="data")
    train_fa = os.path.join("data", "EGCPPIS", "Datasets", "GraphPPIS", "Train_335.fa")
    records = parse_fasta(train_fa)
    print(f"\nLoaded {len(records)} records from Train_335.fa")
    
    # Test protein: 1acbI
    test_record = records[0]
    pid, seq, lbl = test_record
    print(f"Selected protein: {pid}, Length: {len(seq)}, Binding labels: {sum(int(c) for c in lbl)}/{len(lbl)}")
    
    node_feats, dist_mat = loader.load_protein_features(pid, seq)
    assert node_feats is not None and dist_mat is not None, "Failed to load features"
    print(f"Residue features shape: {node_feats.shape} (Expected: ({len(seq)}, 74))")
    print(f"Distance matrix shape: {dist_mat.shape} (Expected: ({len(seq)}, {len(seq)}))")
    
    # Build graph
    edge_index, edge_dists, adj = build_residue_graph(dist_mat, cutoff=16.0)
    print(f"Residue graph: {edge_index.size(1)} edges")

    # 3. Model Architecture & Forward Pass Test
    model = EGCPPISModel(feature_dim=74, num_heads=10).to(device)
    loss_fn = EGCPPISLoss(delta=0.1)

    node_t = torch.tensor(node_feats, device=device)
    coords_t = torch.randn(len(seq), 3, device=device)
    edge_idx_t = edge_index.to(device)
    dist_t = torch.tensor(dist_mat, device=device)
    lbl_t = torch.tensor([int(c) for c in lbl], dtype=torch.float32, device=device).unsqueeze(-1)

    model.eval()
    with torch.no_grad():
        out, cl_loss = model(node_t, coords_t, edge_idx_t, dist_t)
        total_loss, bce, _ = loss_fn(out, lbl_t, cl_loss)

    print(f"Model forward output shape: {out.shape} (Expected: ({len(seq)}, 1))")
    print(f"Initial BCE loss: {bce.item():.4f}, CL loss: {cl_loss.item():.4f}, Total loss: {total_loss.item():.4f}")
    assert not torch.isnan(out).any(), "NaN in model output!"
    assert not torch.isinf(out).any(), "Inf in model output!"
    print("Forward pass VALIDATED successfully!")

    # 4. Tiny Overfit Test (2 proteins, 20 iterations)
    print("\n--- Running Tiny Overfit Test (2 proteins) ---")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    sample_batch = records[:2]
    initial_loss = 0
    final_loss = 0

    for step in range(20):
        step_loss = 0
        optimizer.zero_grad()
        for r_pid, r_seq, r_lbl in sample_batch:
            n_f, d_m = loader.load_protein_features(r_pid, r_seq)
            e_idx, _, _ = build_residue_graph(d_m, cutoff=16.0)
            
            n_t = torch.tensor(n_f, device=device)
            c_t = torch.randn(len(r_seq), 3, device=device)
            e_t = e_idx.to(device)
            d_t = torch.tensor(d_m, device=device)
            y_t = torch.tensor([int(c) for c in r_lbl], dtype=torch.float32, device=device).unsqueeze(-1)
            
            p, cl = model(n_t, c_t, e_t, d_t)
            l, _, _ = loss_fn(p, y_t, cl)
            l.backward()
            step_loss += l.item()
            
        optimizer.step()
        if step == 0:
            initial_loss = step_loss
        if step == 19:
            final_loss = step_loss
        if (step + 1) % 5 == 0:
            print(f"  Iteration {step+1}/20: Loss = {step_loss:.4f}")

    print(f"Initial loss: {initial_loss:.4f} -> Final loss: {final_loss:.4f}")
    assert final_loss < initial_loss, "Overfit test failed to reduce loss!"
    print("Tiny overfit test PASSED! Optimization gradient flows correctly.")

if __name__ == '__main__':
    main()
