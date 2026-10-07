import torch
import torch.nn as nn
import torch.nn.functional as F

class EGCPPISLoss(nn.Module):
    def __init__(self, delta=0.1):
        super(EGCPPISLoss, self).__init__()
        self.delta = delta

    def forward(self, pred, target, cl_loss=None):
        """
        pred: (N, 1) or (1, N)
        target: (N, 1) or (1, N)
        cl_loss: scalar contrastive loss
        """
        pred_clamped = pred.view(-1).clamp(min=1e-7, max=1.0 - 1e-7)
        bce = F.binary_cross_entropy(pred_clamped, target.view(-1).float())
        if cl_loss is not None and cl_loss > 0:
            total_loss = bce + self.delta * cl_loss
        else:
            total_loss = bce
        return total_loss, bce, cl_loss
