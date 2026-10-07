import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

def plot_training_curves(log_csv="results/logs/training_log.csv", out_dir="results/figures"):
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(log_csv):
        print(f"Log file {log_csv} not found.")
        return

    epochs = []
    lrs = []
    tr_losses = []
    val_losses = []
    val_aucs = []
    val_auprs = []
    val_mccs = []

    with open(log_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            lrs.append(float(row['lr']))
            tr_losses.append(float(row['train_loss']))
            val_losses.append(float(row['val_loss']))
            val_aucs.append(float(row['val_auc']))
            val_auprs.append(float(row['val_aupr']))
            val_mccs.append(float(row['val_mcc']))

    if not epochs:
        return

    # 1. Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, tr_losses, label="Train Total Loss", color="#1f77b4", lw=2)
    plt.plot(epochs, val_losses, label="Validation Loss", color="#ff7f0e", lw=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("EGCPPIS Reproduction - Training & Validation Loss", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=300)
    plt.close()

    # 2. Validation Metrics Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_auprs, label="Val AUPR", color="#2ca02c", lw=2)
    plt.plot(epochs, val_aucs, label="Val AUC", color="#d62728", lw=2)
    plt.plot(epochs, val_mccs, label="Val MCC", color="#9467bd", lw=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Metric Score", fontsize=12)
    plt.title("EGCPPIS Reproduction - Validation Performance History", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "validation_metrics.png"), dpi=300)
    plt.close()

    # 3. Learning Rate Curve
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, lrs, label="Learning Rate", color="#e377c2", lw=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("LR", fontsize=12)
    plt.title("Learning Rate Schedule (ReduceLROnPlateau)", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.yscale("log")
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "learning_rate.png"), dpi=300)
    plt.close()
    print("Training curves plotted.")

def plot_benchmark_evaluations(preds_dir="results/metrics", out_dir="results/figures"):
    os.makedirs(out_dir, exist_ok=True)
    datasets = ['Test_60', 'Test_315-28', 'UBtest_31-6', 'Test_70']
    
    # ROC Curves
    plt.figure(figsize=(8, 6))
    for dname in datasets:
        pfile = os.path.join(preds_dir, f"preds_{dname}.npz")
        if os.path.exists(pfile):
            data = np.load(pfile)
            preds = data['preds']
            labels = data['labels']
            fpr, tpr, _ = roc_curve(labels, preds)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{dname} (AUC = {roc_auc:.3f})", lw=2)
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves across Benchmark Datasets", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curves.png"), dpi=300)
    plt.close()

    # PR Curves
    plt.figure(figsize=(8, 6))
    for dname in datasets:
        pfile = os.path.join(preds_dir, f"preds_{dname}.npz")
        if os.path.exists(pfile):
            data = np.load(pfile)
            preds = data['preds']
            labels = data['labels']
            prec, rec, _ = precision_recall_curve(labels, preds)
            pr_auc = auc(rec, prec)
            plt.plot(rec, prec, label=f"{dname} (AUPR = {pr_auc:.3f})", lw=2)
    
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curves across Benchmark Datasets", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pr_curves.png"), dpi=300)
    plt.close()
    print("Benchmark ROC/PR curves plotted.")

if __name__ == '__main__':
    plot_training_curves()
    plot_benchmark_evaluations()
