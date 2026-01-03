
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.methods.base import BaseMethod
from src.methods.dann import grad_reverse

class Disentangled(BaseMethod):
    """
    Proposed Method V2: Disentangled Representation Learning.
    Splits the latent space into:
    - Z_c (Content/Stable): Predictive of Task, Invariant to Domain (Adversarial).
    - Z_s (Style/Spurious): Predictive of Domain (Shortcut encoding).
    
    Loss = L_task(Z_c) + lambda_adv * L_adv(Z_c) + lambda_bias * L_bias(Z_s)
    """
    def __init__(self, model, num_classes, num_domains=2, lambda_adv=1.0, lambda_bias=1.0):
        super(Disentangled, self).__init__(model, num_classes)
        self.num_domains = num_domains
        self.lambda_adv = lambda_adv
        self.lambda_bias = lambda_bias
        
        # 1. Inspect Backbone
        # We assume the model (ResNet1d) has a .fc or output dimension we can split.
        # Actually, ResNet1d.fc is the classifier. We need the feature dim before it.
        # We can hijack `model.fc` again like in DANN, or assume `model` is just the encoder.
        # Let's check `model.fc.in_features`.
        input_dim = model.fc.in_features
        model.fc = nn.Identity() # Remove original head
        
        # 2. Split Dimensionality
        # We split features equally or by ratio? Let's do 50/50 for now.
        self.z_dim = input_dim // 2
        self.s_dim = input_dim - self.z_dim
        
        # 3. Heads
        
        # Task Classifier (Operates on Z_c)
        self.task_classifier = nn.Linear(self.z_dim, num_classes)
        
        # Domain Adversary (Operates on Z_c) - Tries to fail at predicting domain
        self.domain_adversary = nn.Sequential(
            nn.Linear(self.z_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_domains)
        )
        
        # Domain Predictor (Operates on Z_s) - Tries to succeed at predicting domain
        self.domain_predictor = nn.Sequential(
            nn.Linear(self.s_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_domains)
        )

    def forward(self, x):
        # Returns only Task logits for inference
        features = self.model(x)
        z_c = features[:, :self.z_dim]
        return self.task_classifier(z_c)
        
    def training_step(self, batch, batch_idx):
        if len(batch) < 3:
             # Fallback
             x, y = batch[:2]
             d = torch.zeros(len(y), dtype=torch.long, device=x.device)
        else:
             x, y, d = batch
        
        # 1. Encode
        features = self.model(x)
        
        # 2. Split
        z_c = features[:, :self.z_dim]   # Content / Stable
        z_s = features[:, self.z_dim:]   # Style / Spurious
        
        # 3. Task Loss (on Z_c)
        task_logits = self.task_classifier(z_c)
        loss_task = F.cross_entropy(task_logits, y)
        
        # 4. Adversarial Loss (on Z_c)
        # We use Gradient Reversal Layer so that Encoder learns to FOOL the adversary
        z_c_rev = grad_reverse(z_c, alpha=1.0) 
        adv_logits = self.domain_adversary(z_c_rev)
        loss_adv = F.cross_entropy(adv_logits, d)
        
        # 5. Bias/Style Loss (on Z_s)
        # We WANT Z_s to encode domain info (capture the shortcuts here so Z_c doesn't have to)
        # Standard Cross Entropy
        bias_logits = self.domain_predictor(z_s)
        loss_bias = F.cross_entropy(bias_logits, d)
        
        # Total
        loss = loss_task + (self.lambda_adv * loss_adv) + (self.lambda_bias * loss_bias)
        
        # Metrics
        acc_task = (torch.argmax(task_logits, 1) == y).float().mean()
        acc_adv = (torch.argmax(adv_logits, 1) == d).float().mean() # Should go to 0.5 (random guess)
        acc_bias = (torch.argmax(bias_logits, 1) == d).float().mean() # Should go to 1.0 (perfect domain ID)
        
        return {
            'loss': loss,
            'log': {
                'train_loss': loss.item(),
                'train_task_loss': loss_task.item(),
                'train_adv_loss': loss_adv.item(),
                'train_bias_loss': loss_bias.item(),
                'train_acc': acc_task.item(),
                'train_adv_acc': acc_adv.item(),
                'train_bias_acc': acc_bias.item()
            }
        }
    
    def validation_step(self, batch, batch_idx):
        x, y = batch[:2]
        logits = self(x) # Uses Z_c
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        return {
            'loss': loss,
            'preds': preds,
            'targets': y
        }
