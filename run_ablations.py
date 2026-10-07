import os
import json
import csv
import torch
import numpy as np

from src.utils import set_seed, get_device
from src.data_loader import parse_fasta
from src.feature_loader import FeatureLoader
from src.graph_builder import build_residue_graph
from src.model import EGCPPISModel
from src.losses import EGCPPISLoss
from src.metrics import compute_all_metrics

def train_ablation_model(model_name, feature_dim, slice_indices, n_epochs=15, out_dir="results/metrics"):
    set_seed(100000)
    device = get_device()
    os.makedirs(out_dir, exist_ok=True)

    loader = FeatureLoader(data_root="data")
    train_fa = os.path.join("data", "EGCPPIS", "Datasets", "GraphPPIS", "Train_335.fa")
    val_fa = os.path.join("data", "EGCPPIS", "Datasets", "GraphPPIS", "Test_60.fa")

    train_records = parse_fasta(train_fa)
    val_records = parse_fasta(val_fa)

    model = EGCPPISModel(feature_dim=feature_dim, num_heads=10).to(device)
    loss_fn = EGCPPISLoss(delta=0.1 if "No_CL" not in model_name else 0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0008)

    print(f"\n--- Training Ablation: {model_name} (Feature Dim: {feature_dim}, Epochs: {n_epochs}) ---")

    for epoch in range(1, n_epochs + 1):
        model.train()
        for pid, seq, lbl in train_records:
            n_f, d_m = loader.load_protein_features(pid, seq)
            if n_f is None: continue
            if slice_indices is not None:
                n_f = n_f[:, slice_indices]

            e_idx, _, _ = build_residue_graph(d_m, cutoff=16.0)
            n_t = torch.tensor(n_f, device=device)
            c_t = torch.randn(len(seq), 3, device=device)
            e_t = e_idx.to(device)
            d_t = torch.tensor(d_m, device=device)
            y_t = torch.tensor([int(c) for c in lbl], dtype=torch.float32, device=device).unsqueeze(-1)

            optimizer.zero_grad()
            p, cl = model(n_t, c_t, e_t, d_t)
            l, _, _ = loss_fn(p, y_t, cl)
            l.backward()
            optimizer.step()

    # Evaluate on Test_60
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for pid, seq, lbl in val_records:
            n_f, d_m = loader.load_protein_features(pid, seq)
            if n_f is None: continue
            if slice_indices is not None:
                n_f = n_f[:, slice_indices]

            e_idx, _, _ = build_residue_graph(d_m, cutoff=16.0)
            n_t = torch.tensor(n_f, device=device)
            c_t = torch.randn(len(seq), 3, device=device)
            e_t = e_idx.to(device)
            d_t = torch.tensor(d_m, device=device)

            p, _ = model(n_t, c_t, e_t, d_t)
            all_preds.append(p.cpu().numpy().flatten())
            all_labels.append(np.array([int(c) for c in lbl], dtype=np.int32))

    preds_arr = np.concatenate(all_preds)
    labels_arr = np.concatenate(all_labels)
    metrics = compute_all_metrics(preds_arr, labels_arr)
    print(f"  {model_name} Results: AUC={metrics['auc']:.4f}, AUPR={metrics['aupr']:.4f}, MCC={metrics['mcc']:.4f}, F1={metrics['f1']:.4f}")
    return metrics

def run_all_ablations():
    ablations = {
        'Full_Model (74d)': (74, None),
        'Sequence_Only (60d)': (60, list(range(0, 20)) + list(range(34, 74))), # One-hot (20) + PSSM (20) + HMM (20)
        'Structure_Only (14d)': (14, list(range(20, 34))), # DSSP (14)
        'Ablation_No_CL': (74, None),
    }

    ablation_results = {}
    for name, (fdim, sl) in ablations.items():
        m = train_ablation_model(name, fdim, sl, n_epochs=12)
        ablation_results[name] = m

    with open("results/metrics/ablation_results.json", "w") as f:
        json.dump(ablation_results, f, indent=2)

    # Save CSV
    with open("results/metrics/ablation_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Configuration", "Accuracy", "Precision", "Recall", "F1", "MCC", "AUC", "AUPR"])
        for name, res in ablation_results.items():
            writer.writerow([name, round(res['accuracy'], 4), round(res['precision'], 4),
                             round(res['recall'], 4), round(res['f1'], 4), round(res['mcc'], 4),
                             round(res['auc'], 4), round(res['aupr'], 4)])

    print("All ablation studies completed!")

if __name__ == '__main__':
    run_all_ablations()
