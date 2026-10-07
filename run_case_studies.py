import os
import json
import torch
import numpy as np

from src.utils import set_seed, get_device
from src.data_loader import parse_fasta
from src.feature_loader import FeatureLoader
from src.graph_builder import build_residue_graph
from src.model import EGCPPISModel

CASE_PROTEINS = ['4fq0B', '3uvjA', '6kipA', '1t6eX']

def run_case_studies(checkpoint_path="results/checkpoints/best_reproduction_model.pth", out_dir="results/metrics"):
    set_seed(100000)
    device = get_device()
    os.makedirs(out_dir, exist_ok=True)
    loader = FeatureLoader(data_root="data")

    # Find sequences and labels for case proteins across datasets
    all_records = {}
    for dpath in [
        "data/EGCPPIS/Datasets/GraphPPIS/Test_60.fa",
        "data/EGCPPIS/Datasets/GraphPPIS/Test_315-28.fa",
        "data/EGCPPIS/Datasets/GraphPPIS/UBtest_31-6.fa",
        "data/EGCPPIS/Datasets/GraphPPIS/Train_335.fa"
    ]:
        if os.path.exists(dpath):
            for pid, seq, lbl in parse_fasta(dpath):
                all_records[pid.upper()] = (pid, seq, lbl)

    model = EGCPPISModel(feature_dim=74, num_heads=10).to(device)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint for case studies: {checkpoint_path}")

    model.eval()
    case_results = {}

    print("\n" + "="*80)
    print("                      CASE STUDY EVALUATION (PAPER PROTEINS)                    ")
    print("="*80)

    for target_id in CASE_PROTEINS:
        matched = all_records.get(target_id.upper(), None)
        if not matched:
            # Check lowercase
            matched = all_records.get(target_id, None)
        if not matched:
            print(f"Case protein {target_id}: record not found in fasta datasets.")
            continue

        real_pid, seq, lbl = matched
        n_f, d_m = loader.load_protein_features(real_pid, seq)
        if n_f is None:
            # try upper/lower
            n_f, d_m = loader.load_protein_features(target_id, seq)
        if n_f is None:
            print(f"Case protein {target_id}: feature loading failed.")
            continue

        e_idx, _, _ = build_residue_graph(d_m, cutoff=16.0)
        n_t = torch.tensor(n_f, device=device)
        c_t = torch.randn(len(seq), 3, device=device)
        e_t = e_idx.to(device)
        d_t = torch.tensor(d_m, device=device)

        with torch.no_grad():
            p, _ = model(n_t, c_t, e_t, d_t)
            probs = p.cpu().numpy().flatten()

        labels = np.array([int(c) for c in lbl], dtype=np.int32)
        preds = (probs > 0.4).astype(np.int32)

        tp = int(np.sum((preds == 1) & (labels == 1)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))
        tn = int(np.sum((preds == 0) & (labels == 0)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        case_results[target_id] = {
            'protein_id': target_id,
            'length': len(seq),
            'true_binding': int(labels.sum()),
            'predicted_binding': int(preds.sum()),
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'TN': tn,
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1': round(f1, 4),
            'actual_residue_indices': [int(i) for i in np.where(labels == 1)[0]],
            'predicted_residue_indices': [int(i) for i in np.where(preds == 1)[0]]
        }

        print(f"\nProtein {target_id} (Length: {len(seq)}):")
        print(f"  Actual Binding: {labels.sum()} | Predicted Binding: {preds.sum()}")
        print(f"  TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")
        print(f"  Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

    with open(os.path.join(out_dir, "case_studies_results.json"), "w") as f:
        json.dump(case_results, f, indent=2)

    return case_results

if __name__ == '__main__':
    run_case_studies()
