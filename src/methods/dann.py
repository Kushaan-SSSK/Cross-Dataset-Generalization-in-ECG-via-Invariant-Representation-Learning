
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.methods.base import BaseMethod

class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward: identity
    Backward: negate gradients
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def grad_reverse(x, alpha=1.0):
    return GradientReversalFunction.apply(x, alpha)

class DANN(BaseMethod):
    """
    Domain-Adversarial Neural Network (DANN).
    Optimizes for Task Loss - lambda * Domain Loss.
    """
    def __init__(self, model, num_classes, num_domains=2, alpha=1.0):
        super(DANN, self).__init__(model, num_classes)
        self.num_domains = num_domains
        self.alpha = alpha # Gradient reversal strength
        
        # Domain Discriminator (MLP)
        # We assume model output is features before classification? 
        # Wait, ResNet1d forward returns logits. We need features.
        # We might need to modify ResNet to return features OR hook it.
        # For validation, let's assuming we hook or modify.
        # Ideally, 'model' should be 'encoder' + 'classifier'.
        # But 'model' is currently full ResNet.
        
        # HACK: We will assume `model` has `fc` layer and we can tap into `avgpool` or remove `fc`.
        # OR better: The user constructs DANN with `encoder` and `classifier` separately?
        # Given "ResNet1d" is monolithic in `src/models/resnet1d.py`, let's modify it or subclass DANN to generic method.
        
        # For simplicity: We will override `model.fc` with Identity and handle heads here?
        # No, that's messy.
        # Let's assume input model returns features if we set `num_classes=0` or similar?
        # Or we check `model.fc` and replace it.
        
        input_dim = model.fc.in_features
        
        # Task Classifier (Re-create logic of original FC)
        self.classifier = nn.Linear(input_dim, num_classes)
        
        # Domain Discriminator
        self.domain_classifier = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_domains)
        )
        
        # Replace model.fc to avoid running it twice or unused
        model.fc = nn.Identity() 

    def forward(self, x):
        features = self.model(x)
        logits = self.classifier(features)
        return logits

    def training_step(self, batch, batch_idx):
        # We need domain labels for DANN! 
        # Current ECGDataset returns (x, y). We need (x, y, d).
        # We need to update Dataset to return domain 'dataset_source'.
        
        # NOTE: For now, avoiding code break, we assume batch has 3 elements if we modify dataset
        # OR we just use dummy for implementation and update dataset next.
        
        x, y = batch[:2]
        d = batch[2] if len(batch) > 2 else torch.zeros(len(y), dtype=torch.long, device=x.device) # Dummy domain
        
        # Features
        features = self.model(x)
        
        # Task Loss
        class_logits = self.classifier(features)
        class_loss = F.cross_entropy(class_logits, y)
        
        # Domain Loss with Gradient Reversal
        features_rev = grad_reverse(features, self.alpha)
        domain_logits = self.domain_classifier(features_rev)
        domain_loss = F.cross_entropy(domain_logits, d)
        
        # Total Loss
        loss = class_loss + domain_loss
        
        # Metrics
        class_acc = (torch.argmax(class_logits, 1) == y).float().mean()
        domain_acc = (torch.argmax(domain_logits, 1) == d).float().mean()
        
        return {
            'loss': loss,
            'log': {
                'train_loss': loss.item(), 
                'train_class_loss': class_loss.item(),
                'train_domain_loss': domain_loss.item(),
                'train_acc': class_acc.item(),
                'domain_acc': domain_acc.item()
            }
        }

    def validation_step(self, batch, batch_idx):
        x, y = batch[:2]
        features = self.model(x)
        logits = self.classifier(features)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        return {
            'loss': loss,
            'preds': preds,
            'targets': y
        }
