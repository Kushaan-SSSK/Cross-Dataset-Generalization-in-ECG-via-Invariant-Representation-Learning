
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
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        return {
            'loss': loss,
            'preds': preds,
            'targets': y
        }
