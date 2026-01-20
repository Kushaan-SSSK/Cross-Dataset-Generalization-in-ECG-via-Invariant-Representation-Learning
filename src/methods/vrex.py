
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.methods.base import BaseMethod

class VREx(BaseMethod):
    """
    V-REx: Variance Risk Extrapolation.
    Penalizes the variance of risks (losses) across domains to encourage invariance.
    Loss = AvgLoss + beta * Var(Losses)
    Supports optional class weighting for imbalanced datasets.
    """
    def __init__(self, model, num_classes, beta=10.0, annealing_epochs=10, class_weights=None):
        super(VREx, self).__init__(model, num_classes)
        self.beta = beta
        self.annealing_epochs = annealing_epochs
        # We need to track epochs for annealing if we want to delay the penalty
        self.register_buffer('steps_counter', torch.tensor(0))
        
        # Class weights for imbalanced data
        if class_weights is not None:
            if not isinstance(class_weights, torch.Tensor):
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch[:2]
        # Domain is needed!
        if len(batch) < 3:
             # Fallback if no domain, generic ERM behavior (variance is 0)
             # But our dataset gives it now.
             domains = torch.zeros(len(y), device=x.device)
        else:
             domains = batch[2]

        logits = self(x)
        
        # We need to compute loss PER domain
        unique_domains = torch.unique(domains)
        losses = []
        
        # Calculate loss for each domain presence in the batch
        for d in unique_domains:
            mask = (domains == d)
            if mask.sum() > 0:
                domain_loss = F.cross_entropy(logits[mask], y[mask], weight=self.class_weights)
                losses.append(domain_loss)
        
        if len(losses) > 1:
            losses_tensor = torch.stack(losses)
            mean_loss = losses_tensor.mean()
            # Unbiased variance requires N > 1
            if losses_tensor.size(0) > 1:
                var_loss = losses_tensor.var()
            else:
                 var_loss = torch.tensor(0.0, device=x.device)
        else:
            # Only one domain in batch -> No variance penalty
            if len(losses) == 1:
                mean_loss = losses[0]
            else:
                # No domains found? (Empty batch or filtering issue)
                mean_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
            
            var_loss = torch.tensor(0.0, device=x.device)

        # Beta Annealing (optional, often used to let model learn feature first)
        # Assuming we just use constant beta for simplicity unless specified
        
        total_loss = mean_loss + self.beta * var_loss

        # Metrics
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        return {
            'loss': total_loss,
            'log': {
                'train_loss': total_loss.item(), 
                'train_mean_loss': mean_loss.item(),
                'train_var_loss': var_loss.item(),
                'train_acc': acc.item()
            }
        }

    def validation_step(self, batch, batch_idx):
        x, y = batch[:2]
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        return {
            'loss': loss,
            'preds': preds,
            'targets': y
        }
