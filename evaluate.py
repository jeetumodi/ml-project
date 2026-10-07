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

PAPER_TARGETS = {
    'Test_60': {
        'accuracy': 0.873,
        'precision': 0.584,
        'recall': 0.668,
        'f1': 0.623,
        'mcc': 0.549,
        'auc': 0.890,
        'aupr': 0.656
    },
    'Test_315-28': {
        'mcc': 0.511,
        'auc': 0.885,
        'aupr': 0.595
    },
    'UBtest_31-6': {
        'mcc': 0.401,
        'auc': 0.845,
        'aupr': 0.445
    },
    'Test_70': {
        'accuracy': 0.851,
        'precision': 0.621,
        'recall': 0.639,
        'f1': 0.630,
        'mcc': 0.537,
        'auc': 0.880,
        'aupr': 0.682
    }
}

def evaluate_on_records(model, records, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    evaluated_pids = []

    with torch.no_grad():
        for pid, seq, lbl in records:
            n_f, d_m = loader.load_protein_features(pid, seq)
            if n_f is None:
                continue
            e_idx, _, _ = build_residue_graph(d_m, cutoff=16.0)

            n_t = torch.tensor(n_f, device=device)
            c_t = torch.randn(len(seq), 3, device=device)
            e_t = e_idx.to(device)
            d_t = torch.tensor(d_m, device=device)

            p, _ = model(n_t, c_t, e_t, d_t)
            all_preds.append(p.cpu().numpy().flatten())
            all_labels.append(np.array([int(c) for c in lbl], dtype=np.int32))
            evaluated_pids.append(pid)

    if not all_preds:
        return None, None, None

    preds_arr = np.concatenate(all_preds)
    labels_arr = np.concatenate(all_labels)
    metrics = compute_all_metrics(preds_arr, labels_arr)
    return metrics, preds_arr, labels_arr

def run_evaluation(checkpoint_path="results/checkpoints/best_reproduction_model.pth",
                   out_dir="results/metrics"):
    set_seed(100000)
    device = get_device()
    os.makedirs(out_dir, exist_ok=True)

    loader = FeatureLoader(data_root="data")
    datasets = {
        'Test_60': os.path.join("data", "EGCPPIS", "Datasets", "GraphPPIS", "Test_60.fa"),
        'Test_315-28': os.path.join("data", "EGCPPIS", "Datasets", "GraphPPIS", "Test_315-28.fa"),
        'UBtest_31-6': os.path.join("data", "EGCPPIS", "Datasets", "GraphPPIS", "UBtest_31-6.fa"),
        'Test_70': os.path.join("data", "EGCPPIS", "Datasets", "DeepPPISP", "Test_70.fa"),
    }

    model = EGCPPISModel(feature_dim=74, num_heads=10).to(device)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint from: {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Evaluating untrained weights.")

    results = {}
    comparison_table = []

    print("\n" + "="*85)
    print("                      BENCHMARK DATASET EVALUATION RESULTS                   ")
    print("="*85)

    for dname, dpath in datasets.items():
        records = parse_fasta(dpath)
        metrics, preds, labels = evaluate_on_records(model, records, loader, device)
        if metrics is None:
            print(f"Dataset {dname}: No valid proteins could be evaluated.")
            continue

        results[dname] = metrics
        print(f"\nDataset: {dname} (Total Evaluated Residues: {len(labels):,}, Positives: {labels.sum():,})")
        for mkey in ['accuracy', 'precision', 'recall', 'f1', 'mcc', 'auc', 'aupr']:
            mval = metrics.get(mkey, 0.0)
            paper_val = PAPER_TARGETS.get(dname, {}).get(mkey, None)
            diff_str = f"{(mval - paper_val):+.3f}" if paper_val is not None else "N/A"
            paper_str = f"{paper_val:.3f}" if paper_val is not None else "N/A"
            print(f"  {mkey:<12s}: Repro = {mval:.3f} | Paper = {paper_str:>5s} | Diff = {diff_str}")

            comparison_table.append({
                'dataset': dname,
                'metric': mkey,
                'paper': paper_val if paper_val is not None else "",
                'reproduction': round(mval, 4),
                'difference': round(mval - paper_val, 4) if paper_val is not None else ""
            })

        # Save predictions
        np.savez_compressed(os.path.join(out_dir, f"preds_{dname}.npz"), preds=preds, labels=labels)

    # Save comparison table
    csv_path = os.path.join(out_dir, "benchmark_comparison.csv")
    with open(csv_path, "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=['dataset', 'metric', 'paper', 'reproduction', 'difference'])
        writer.writeheader()
        writer.writerows(comparison_table)

    json_path = os.path.join(out_dir, "benchmark_results.json")
    with open(json_path, "w") as jf:
        json.dump(results, jf, indent=2)

    print("\nEvaluation results saved to:")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    return results

if __name__ == '__main__':
    run_evaluation()
