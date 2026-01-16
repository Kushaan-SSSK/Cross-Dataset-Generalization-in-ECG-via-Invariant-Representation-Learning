
import numpy as np
from omegaconf import OmegaConf
import logging

log = logging.getLogger(__name__)

class SASTProtocol:
    """
    Synthetic Artifact Shortcut Task (SAST) Protocol.
    Manages configuration, injection logic, and summary logging for shortcut artifacts.
    """
    def __init__(self, cfg):
        """
        Args:
            cfg (DictConfig or dict): Configuration containing:
                - use_shortcut (bool): Master toggle
                - type (str): 'mains', 'bw', 'emg'
                - correlation (float): Correlation strength (0.0 to 1.0)
                - force (bool): If True, injects 100% of the time (for testing/poisoned sets)
                - amplitude (float): Amplitude of artifact
                - freq (float): Frequency (for mains)
                - split (str): 'train', 'val', 'test'
        """
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        self.cfg = cfg
        self.use_shortcut = OmegaConf.select(cfg, "use_shortcut", default=False)
        self.type = OmegaConf.select(cfg, "type", default="mains")
        self.correlation = OmegaConf.select(cfg, "correlation", default=0.0)
        self.force = OmegaConf.select(cfg, "force", default=False)
        self.amplitude = OmegaConf.select(cfg, "amplitude", default=0.1)
        self.freq = OmegaConf.select(cfg, "freq", default=60)
        self.split = OmegaConf.select(cfg, "split", default="train")
        
        # Log Summary on Init
        if self.use_shortcut:
            log.info(self.get_summary())

    def get_summary(self):
        if not self.use_shortcut:
            return "SAST Protocol: Disabled"
        
        mode = "Forced (100%)" if self.force else f"Correlated (Rho={self.correlation})"
        return f"SAST Protocol: Type={self.type}, Amp={self.amplitude}, Mode={mode}, Split={self.split}"

    def inject_batch(self, signals, labels=None, idxs=None):
        """
        Injects artifacts into a batch of signals.
        Args:
            signals (np.ndarray): (B, C, L)
            labels (np.ndarray, optional): Labels for correlation logic.
            idxs (np.ndarray, optional): Sample indices for deterministic RNG.
        Returns:
            np.ndarray: Modified signals.
        """
        if not self.use_shortcut:
            return signals
            
        B, C, L = signals.shape
        fs = 100 # Assumed
        t = np.arange(L) / fs
        
        # Determine mask of samples to inject
        inject_mask = np.zeros(B, dtype=bool)
        
        if self.force:
            inject_mask[:] = True
        elif self.split == 'train' and labels is not None:
             # Correlation Logic
             # Target Class: We assume Class 1 (AFIB) is the target for shortcut?
             # Or generic "Abnormal"?
             # Prompt 8 says "verify poisoning efficacy".
             # We need a clear definition. 
             # Let's define "Target" as Class > 0 (Abnormal) for now, or Class 1.
             # Given 7-class, let's target Class 1 (Atrial Fibrillation) specifically for visual clarity?
             # Or target ALL abnormal (1-6)?
             # Reverting to Binary concept: Abnormal (1-6) vs Normal (0).
             # Correlation p means: P(Artifact | Abnormal) = p, P(Artifact | Normal) = 1-p ?
             # Or P(Artifact | Abnormal) = p, P(Artifact | Normal) = 0? (Single bit)
             # Standard Spurious Corr: 
             # Group 0 (Normal): 10% Noise (if rho=0.9)
             # Group 1 (Abnormal): 90% Noise (if rho=0.9)
             
             # Let's implement: IsAbnormal = (label != 0)
             # If label != 0: Prob = correlation
             # If label == 0: Prob = 1 - correlation
             
             is_abnormal = (labels != 0)
             rng = np.random.default_rng(seed=42) # Should use idxs for true determinism
             
             probs = np.where(is_abnormal, self.correlation, 1 - self.correlation)
             
             # If idxs provided, use them for seed
             if idxs is not None:
                 for i, seed in enumerate(idxs):
                     # Unique seed per sample
                     r = np.random.default_rng(seed=seed).random()
                     inject_mask[i] = r < probs[i]
             else:
                 inject_mask = rng.random(B) < probs
        else:
            # Valid/Test split without Force: Do NOT inject (Clean evaluation)
            inject_mask[:] = False
            
        # Optimization: Apply noise only to masked
        indices = np.where(inject_mask)[0]
        if len(indices) == 0:
            return signals
            
        # Generate Noise
        # Vectorized generation is hard with variable noise (EMG).
        # We'll loop for safety or use simple vectorization for Mains.
        
        if self.type == 'mains':
            noise = self.amplitude * np.sin(2 * np.pi * self.freq * t)
            # Add to Lead I (Channel 0)
            signals[indices, 0, :] += noise
            
        elif self.type == 'bw':
            noise = self.amplitude * np.sin(2 * np.pi * 0.5 * t) + (self.amplitude * 0.5 * t)
            signals[indices, 0, :] += noise
            
        elif self.type == 'emg':
            # EMG needs randomness per sample
            for i in indices:
                seed = int(signals[i, 0, 0]*1000) if idxs is None else idxs[i]
                rng = np.random.default_rng(seed=seed)
                mask = rng.random(L) > 0.5
                raw_noise = rng.normal(0, 1, L)
                noise = self.amplitude * raw_noise * mask
                signals[i, 0, :] += noise
                
        return signals

    def inject_single(self, signal, label=None, unique_id=None):
        """
        Inject into a single sample (for Dataset __getitem__).
        """
        # Wrap as batch
        # Signal: (C, L) -> (1, C, L)
        sig_batch = signal[np.newaxis, ...]
        
        lbl_batch = None
        if label is not None:
            lbl_batch = np.array([label])
            
        idxs = None
        if unique_id is not None:
             # Hash unique_id to int
             idxs = np.array([hash(unique_id) % (2**32)])
             
        res = self.inject_batch(sig_batch, labels=lbl_batch, idxs=idxs)
        return res[0]
