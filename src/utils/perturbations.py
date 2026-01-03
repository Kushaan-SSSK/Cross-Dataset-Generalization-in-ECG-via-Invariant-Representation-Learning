
import torch
import numpy as np

class Perturbations:
    """
    Suite of ECG signal perturbations for stress testing robustness.
    Input: (B, C, L) Tensor
    """
    
    @staticmethod
    def add_noise(x, noise_level=0.1):
        """
        Add Gaussian noise.
        noise_level: std dev relative to signal scale (approx 1.0 after Z-score)
        """
        noise = torch.randn_like(x) * noise_level
        return x + noise

    @staticmethod
    def baseline_wander(x, frequency=0.5, amplitude=0.5, fs=100):
        """
        Add low-frequency sinusoidal wander.
        frequency: Hz (wander is usually < 0.5 Hz)
        """
        B, C, L = x.shape
        t = torch.linspace(0, L/fs, L, device=x.device)
        # Create wander: (L,) -> (1, 1, L)
        wander = amplitude * torch.sin(2 * np.pi * frequency * t)
        wander = wander.view(1, 1, -1).expand(B, C, -1)
        return x + wander

    @staticmethod
    def amplitude_scale(x, scale_factor=1.0):
        """
        Scale signal amplitude.
        scale_factor: <1 (attenuation), >1 (amplification)
        """
        return x * scale_factor

    @staticmethod
    def lead_dropout(x, drop_prob=0.5):
        """
        Randomly zero out leads.
        drop_prob: Probability of EACH lead being dropped INDEPENDENTLY.
        """
        B, C, L = x.shape
        # Mask shape: (B, C, 1)
        mask = torch.bernoulli(torch.full((B, C, 1), 1 - drop_prob, device=x.device))
        return x * mask

    @staticmethod
    def apply_perturbation(x, perturbation_name, level):
        """
        Factory method to apply a specific perturbation.
        """
        if perturbation_name == 'noise':
            return Perturbations.add_noise(x, noise_level=level)
        elif perturbation_name == 'wander':
            return Perturbations.baseline_wander(x, amplitude=level)
        elif perturbation_name == 'scale':
            return Perturbations.amplitude_scale(x, scale_factor=level)
        elif perturbation_name == 'dropout':
            # For dropout, 'level' can be drop_prob
            return Perturbations.lead_dropout(x, drop_prob=level)
        else:
            raise ValueError(f"Unknown perturbation: {perturbation_name}")

