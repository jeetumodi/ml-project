import numpy as np
from sklearn.metrics import roc_curve, auc, matthews_corrcoef, precision_recall_curve, accuracy_score

def compute_roc(preds, labels):
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc

def compute_aupr(preds, labels):
    p, r, _ = precision_recall_curve(labels.flatten(), preds.flatten())
    aupr_score = auc(r, p)
    return aupr_score, p, r

def compute_mcc(preds, labels, threshold=0.5):
    bin_preds = (preds > threshold).astype(np.int32)
    return matthews_corrcoef(labels.flatten(), bin_preds.flatten())

def compute_acc(preds, labels, threshold=0.5):
    bin_preds = (preds > threshold).astype(np.int32)
    return accuracy_score(labels.flatten(), bin_preds.flatten())

def compute_all_metrics(preds, labels):
    """
    Computes Accuracy, Precision, Recall, F1, MCC, AUC, AUPR over optimal threshold and standard threshold.
    """
    labels = labels.flatten()
    preds = preds.flatten()
    
    # 1. AUC and AUPR
    roc_auc = compute_roc(preds, labels)
    aupr_score, p_curve, r_curve = compute_aupr(preds, labels)
    
    # 2. Threshold search for optimal F1
    best_f1 = 0.0
    best_p = 0.0
    best_r = 0.0
    best_thresh = 0.5
    best_mcc = 0.0
    best_acc = 0.0
    
    for t in range(1, 100):
        thresh = t / 100.0
        bin_p = (preds > thresh).astype(np.int32)
        tp = np.sum((bin_p == 1) & (labels == 1))
        fp = np.sum((bin_p == 1) & (labels == 0))
        fn = np.sum((bin_p == 0) & (labels == 1))
        tn = np.sum((bin_p == 0) & (labels == 0))
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        if f1 > best_f1:
            best_f1 = f1
            best_p = prec
            best_r = rec
            best_thresh = thresh
            best_acc = (tp + tn) / len(labels)
            best_mcc = matthews_corrcoef(labels, bin_p) if (tp + fp > 0 and fn + tn > 0) else 0.0
            
    return {
        'accuracy': best_acc,
        'precision': best_p,
        'recall': best_r,
        'f1': best_f1,
        'mcc': best_mcc,
        'auc': roc_auc,
        'aupr': aupr_score,
        'threshold': best_thresh
    }
