
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.methods.base import BaseMethod
import torch.autograd as autograd

class IRM(BaseMethod):
    """
    Invariant Risk Minimization (IRMv1).
    Arjovsky et al., 2019.
    
    Minimizes: Sum(Risk_e) + lambda * Sum(|Grad(Risk_e, w=1)|)
    Requires a 'dummy' classifier on top of the representation (scale=1.0 fixed).
    """
    def __init__(self, model, num_classes, penalty_weight=100.0, annealing_epochs=10):
        super(IRM, self).__init__(model, num_classes)
        self.penalty_weight = penalty_weight
        self.annealing_epochs = annealing_epochs
        # We assume the model provided is the full ResNet1d.
        # IRM technically requires splitting into Representation (Phi) and Classifier (w).
        # We can implement this by intercepting the forward pass or using a hook.
        
        # Similar to DANN, we will assume model.fc is the classifier.
        # But IRMv1 formulation usually optimizes a FIXED scalar dummy classifier? 
        # No, that's just the derivation. In practice, 'w' is the final linear layer.
        # We penalized the gradient of the loss w.r.t a dummy scale factor 'w=1.0' applied to the logits.
        
        # We assume 'model' already has a classifier at the end.
        # We just need to compute the gradient penalty efficiently.
        
    def forward(self, x):
        return self.model(x)

    def irm_penalty(self, logits, y):
        device = logits.device
        # Create a dummy scale parameter (w=1.0)
        scale = torch.tensor(1.0, device=device, requires_grad=True)
        
        # Loss using the scaled logits
        loss = F.cross_entropy(logits * scale, y)
        
        # Compute gradient of loss w.r.t. scale
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        
        # Penalty is the squared L2 norm of that gradient
        return torch.sum(grad**2)

    def training_step(self, batch, batch_idx):
        x, y = batch[:2]
        
        # Domain is needed!
        if len(batch) < 3:
             if batch_idx == 0:
                 print("Warning: IRM logic receiving batches without domain info. Defaulting to standard ERM behavior.")
             domains = torch.zeros(len(y), device=x.device)
        else:
             domains = batch[2]

        # 1. Forward Pass to get Logits
        logits = self.model(x)
        
        # 2. Compute Loss & Penalty per Environment
        unique_domains = torch.unique(domains)
        losses = []
        penalties = []
        
        for d in unique_domains:
            mask = (domains == d)
            if mask.sum() > 0:
                d_logits = logits[mask]
                d_y = y[mask]
                
                # Standard Risk
                d_loss = F.cross_entropy(d_logits, d_y)
                losses.append(d_loss)
                
                # IRM Penalty
                d_penalty = self.irm_penalty(d_logits, d_y)
                penalties.append(d_penalty)
        
        # Aggregate
        if len(losses) > 0:
            mean_loss = torch.stack(losses).mean()
            mean_penalty = torch.stack(penalties).mean()
        else:
            mean_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
            mean_penalty = torch.tensor(0.0, device=x.device)
            
        # Annealing (Optional: heavily recommended for IRM to work)
        # We just assume weight is applied fully or we could track steps.
        # For this implementation, we apply fully to be consistent with config.
        # (Or user handles annealing in config, but we have no scheduler hook here easily)
        
        total_loss = mean_loss + self.penalty_weight * mean_penalty
        
        # Metrics
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        return {
            'loss': total_loss,
            'log': {
                'train_loss': total_loss.item(),
                'train_irm_risk': mean_loss.item(),
                'train_irm_penalty': mean_penalty.item(),
                'train_acc': acc.item()
            }
        }

    def validation_step(self, batch, batch_idx):
        x, y = batch[:2]
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        return {
            'loss': loss,
            'preds': preds,
            'targets': y
        }
