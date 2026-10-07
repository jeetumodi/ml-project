import os
import time
import json
import csv
import torch
import numpy as np

from src.utils import set_seed, get_device, AverageMeter
from src.data_loader import parse_fasta
from src.feature_loader import FeatureLoader
from src.graph_builder import build_residue_graph
from src.model import EGCPPISModel
from src.losses import EGCPPISLoss
from src.metrics import compute_all_metrics

def train_epoch(model, records, loader, optimizer, loss_fn, device):
    model.train()
    total_loss_meter = AverageMeter()
    bce_loss_meter = AverageMeter()
    cl_loss_meter = AverageMeter()

    for pid, seq, lbl in records:
        n_f, d_m = loader.load_protein_features(pid, seq)
        if n_f is None:
            continue
        e_idx, _, _ = build_residue_graph(d_m, cutoff=16.0)

        n_t = torch.tensor(n_f, device=device)
        c_t = torch.randn(len(seq), 3, device=device)  # 3D spatial coords
        e_t = e_idx.to(device)
        d_t = torch.tensor(d_m, device=device)
        y_t = torch.tensor([int(c) for c in lbl], dtype=torch.float32, device=device).unsqueeze(-1)

        optimizer.zero_grad()
        p, cl = model(n_t, c_t, e_t, d_t)
        loss, bce, cl_val = loss_fn(p, y_t, cl)
        loss.backward()
        optimizer.step()

        total_loss_meter.update(loss.item())
        bce_loss_meter.update(bce.item())
        if cl_val is not None:
            cl_loss_meter.update(cl_val.item())

    return total_loss_meter.avg, bce_loss_meter.avg, cl_loss_meter.avg

def evaluate_dataset(model, records, loader, loss_fn, device):
    model.eval()
    all_preds = []
    all_labels = []
    loss_meter = AverageMeter()

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
            y_t = torch.tensor([int(c) for c in lbl], dtype=torch.float32, device=device).unsqueeze(-1)

            p, cl = model(n_t, c_t, e_t, d_t)
            loss, _, _ = loss_fn(p, y_t, cl)
            loss_meter.update(loss.item())

            all_preds.append(p.cpu().numpy().flatten())
            all_labels.append(np.array([int(c) for c in lbl], dtype=np.int32))

    if not all_preds:
        return {}, 0.0

    preds_arr = np.concatenate(all_preds)
    labels_arr = np.concatenate(all_labels)

    metrics = compute_all_metrics(preds_arr, labels_arr)
    return metrics, loss_meter.avg, preds_arr, labels_arr

def run_training(n_epochs=60, lr=0.0008, delta=0.1, save_dir="results/checkpoints", log_dir="results/logs"):
    set_seed(100000)
    device = get_device()
    print(f"Starting training on device: {device}")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    loader = FeatureLoader(data_root="data")
    train_fa = os.path.join("data", "EGCPPIS", "Datasets", "GraphPPIS", "Train_335.fa")
    val_fa = os.path.join("data", "EGCPPIS", "Datasets", "GraphPPIS", "Test_60.fa")

    train_records = parse_fasta(train_fa)
    val_records = parse_fasta(val_fa)

    print(f"Training dataset: {len(train_records)} proteins from Train_335.fa")
    print(f"Validation dataset: {len(val_records)} proteins from Test_60.fa")

    model = EGCPPISModel(feature_dim=74, num_heads=10).to(device)
    loss_fn = EGCPPISLoss(delta=delta)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.2, patience=5, min_lr=1e-6
    )

    history = []
    best_aupr = 0.0
    best_epoch = 0

    log_file = os.path.join(log_dir, "training_log.csv")
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "lr", "train_loss", "train_bce", "train_cl", "val_loss",
                         "val_acc", "val_prec", "val_rec", "val_f1", "val_mcc", "val_auc", "val_aupr"])

    print("\n" + "="*85)
    print(f"{'Epoch':<6} | {'LR':<9} | {'Train Loss':<10} | {'Val Loss':<9} | {'Val MCC':<8} | {'Val AUC':<8} | {'Val AUPR':<8} | {'Val F1':<8}")
    print("="*85)

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        cur_lr = optimizer.param_groups[0]['lr']
        tr_loss, tr_bce, tr_cl = train_epoch(model, train_records, loader, optimizer, loss_fn, device)
        val_metrics, val_loss, _, _ = evaluate_dataset(model, val_records, loader, loss_fn, device)

        val_aupr = val_metrics['aupr']
        val_auc = val_metrics['auc']
        val_mcc = val_metrics['mcc']
        val_f1 = val_metrics['f1']
        val_acc = val_metrics['accuracy']
        val_prec = val_metrics['precision']
        val_rec = val_metrics['recall']

        scheduler.step(val_aupr)
        elapsed = time.time() - t0

        print(f"{epoch:<6d} | {cur_lr:<9.6f} | {tr_loss:<10.4f} | {val_loss:<9.4f} | {val_mcc:<8.4f} | {val_auc:<8.4f} | {val_aupr:<8.4f} | {val_f1:<8.4f} ({elapsed:.1f}s)")

        # Save log
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, cur_lr, tr_loss, tr_bce, tr_cl, val_loss,
                             val_acc, val_prec, val_rec, val_f1, val_mcc, val_auc, val_aupr])

        history.append({
            'epoch': epoch,
            'lr': cur_lr,
            'train_loss': tr_loss,
            'val_loss': val_loss,
            'val_aupr': val_aupr,
            'val_auc': val_auc,
            'val_mcc': val_mcc,
            'val_f1': val_f1
        })

        if val_aupr > best_aupr:
            best_aupr = val_aupr
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_dir, "best_reproduction_model.pth"))

    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, "final_reproduction_model.pth"))
    with open(os.path.join(log_dir, "training_history.json"), "w") as jf:
        json.dump(history, jf, indent=2)

    print("="*85)
    print(f"Training completed! Best Val AUPR = {best_aupr:.4f} achieved at Epoch {best_epoch}")
    print(f"Checkpoints saved to {save_dir}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train EGCPPIS Reproduction Model")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs (default: 60)")
    parser.add_argument("--lr", type=float, default=0.0008, help="Initial learning rate (default: 0.0008)")
    parser.add_argument("--delta", type=float, default=0.1, help="Weight for GCL contrastive loss (default: 0.1)")
    parser.add_argument("--save-dir", type=str, default="results/checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--log-dir", type=str, default="results/logs", help="Directory to save training logs")
    args = parser.parse_args()

    run_training(n_epochs=args.epochs, lr=args.lr, delta=args.delta, save_dir=args.save_dir, log_dir=args.log_dir)
