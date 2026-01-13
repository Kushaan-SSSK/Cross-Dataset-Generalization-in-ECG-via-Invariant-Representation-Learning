
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.methods.base import BaseMethod

class ERM(BaseMethod):
    """
    Standard Empirical Risk Minimization.
    """
    def __init__(self, model, num_classes):
        super(ERM, self).__init__(model, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch[:2] # Ignore domain if present
        logits = self(x)
        
        # --- SAFETY CHECK: labels must be int64 in [0, C-1] for CrossEntropyLoss ---
        if y is None:
            raise ValueError("y is None in training_step")
        if y.dtype != torch.long:
            y = y.long()
        # y should be shape [B]
        if y.ndim != 1:
            raise ValueError(f"Expected y.ndim==1, got {y.ndim}, y.shape={tuple(y.shape)}")
        C = logits.shape[1]
        ymin = int(y.min().item())
        ymax = int(y.max().item())
        if ymin < 0 or ymax >= C:
            raise ValueError(
                f"Invalid label range for CrossEntropyLoss: min={ymin}, max={ymax}, "
                f"num_classes(C)={C}, logits.shape={tuple(logits.shape)}, y.shape={tuple(y.shape)}"
            )
        # -------------------------------------------------------------------------

        loss = self.criterion(logits, y)
        
        # Calculate acc for monitoring
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        return {
            'loss': loss,
            'log': {'train_loss': loss.item(), 'train_acc': acc.item()}
        }

    def validation_step(self, batch, batch_idx):
        x, y = batch[:2]
        logits = self(x)
        
        # --- SAFETY CHECK: labels must be int64 in [0, C-1] for CrossEntropyLoss ---
        if y is None:
            raise ValueError("y is None in training_step")
        if y.dtype != torch.long:
            y = y.long()
        # y should be shape [B]
        if y.ndim != 1:
            raise ValueError(f"Expected y.ndim==1, got {y.ndim}, y.shape={tuple(y.shape)}")
        C = logits.shape[1]
        ymin = int(y.min().item())
        ymax = int(y.max().item())
        if ymin < 0 or ymax >= C:
            raise ValueError(
                f"Invalid label range for CrossEntropyLoss: min={ymin}, max={ymax}, "
                f"num_classes(C)={C}, logits.shape={tuple(logits.shape)}, y.shape={tuple(y.shape)}"
            )
        # -------------------------------------------------------------------------

        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        return {
            'loss': loss,
            'preds': preds,
            'targets': y
        }
