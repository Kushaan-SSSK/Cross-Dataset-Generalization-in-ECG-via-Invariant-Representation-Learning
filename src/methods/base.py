
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseMethod(nn.Module, ABC):
    """
    Abstract Base Class for all training methods (ERM, DANN, etc.).
    Wraps the backbone model and defines the training logic.
    """
    def __init__(self, model, num_classes):
        super(BaseMethod, self).__init__()
        self.model = model
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input batch (B, C, L)
        Returns:
            torch.Tensor: Logits (B, num_classes)
        """
        pass

    @abstractmethod
    def training_step(self, batch, batch_idx):
        """
        Single training step.
        Args:
            batch: tuple of (x, y)
            batch_idx: int
        Returns:
            dict: {'loss': tensor, 'log': dict}
        """
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        """
        Single validation step.
        Args:
            batch: tuple of (x, y)
            batch_idx: int
        Returns:
            dict: {'loss': tensor, 'preds': tensor, 'targets': tensor}
        """
        pass

    def configure_optimizers(self, lr, weight_decay):
        """
        Configure optimizers.
        """
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
