
import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import json
import logging

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import ECGDataset
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def main():
    # Hardcoded paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    manifest_path = os.path.join(base_dir, "data", "manifests", "master_manifest.csv")
    split_path = os.path.join(base_dir, "data", "manifests", "splits.json")
    processed_path = os.path.join(base_dir, "data", "processed", "signals.h5") 
    
    # Check paths
    if not os.path.exists(manifest_path):
        print(f"ERROR: {manifest_path} not found.")
        return

    # Load Data Frame
    manifest_df = pd.read_csv(manifest_path)
    print(f"Manifest loaded. Columns: {manifest_df.columns.tolist()}")
    
    # Check sampling rate column
    if 'sampling_rate' in manifest_df.columns:
        sr_counts = manifest_df['sampling_rate'].value_counts()
        print(f"Sampling Rates in Manifest:\n{sr_counts}")
    
    # Setup Dataset (Poisoned Config)
    # Force injection on everything (correlation=1.0)
    sc_cfg = OmegaConf.create({"use_shortcut": True, "freq": 60, "amplitude": 1.0, "correlation": 1.0})
    
    # Create dataset
    # We use split='train' to enable injection logic in __getitem__
    # We use a small subset
    ds = ECGDataset(manifest_df.head(10), processed_path, task_label_col="task_a_label", shortcut_cfg=sc_cfg, split='train')
    
    # Get a sample
    print("\n--- Inspecting Sample 0 ---")
    x, y, _ = ds[0] # (C, L) Correctly unpacking 3 values
    print(f"Shape: {x.shape}")
    print(f"Max Val: {x.max()}, Min Val: {x.min()}")
    
    sig = x[0, :].numpy()
    L = len(sig)
    fs = 100 # Assumed
    
    # Check if injection happened
    # We can't easily know if this SPECIFIC sample got injected unless we control RNG.
    # But correlation=1.0 should inject on everything? 
    # Wait, dataset logic:
    # is_abnormal = (label != 0)
    # if is_abnormal: if rng < p_corr: inject
    # else: if rng < (1-p_corr): inject
    # labels are strings or ints? 
    # If correlation=1.0:
    # Abnormal -> 100% inject
    # Normal -> 0% inject
    # So if sample 0 is Normal, it won't get injected!
    
    label_val = y.item()
    print(f"Label: {label_val}")
    
    # Let's manually inject to be sure what 60Hz looks like
    t = np.linspace(0, L/fs, L, endpoint=False)
    manual_noise = np.sin(2 * np.pi * 60 * t)
    sig_manual = sig + manual_noise
    
    # Compute PSD of Manual injection
    freqs, psd = signal.welch(sig_manual, fs=fs, nperseg=256)
    
    peak_freq = freqs[np.argmax(psd)]
    print(f"Peak Frequency of Manually Injected 60Hz: {peak_freq} Hz")
    
    if abs(peak_freq - 40.0) < 5:
        print(">> CONFIRMED: 60Hz aliases to ~40Hz at fs=100Hz.")
    elif abs(peak_freq - 60.0) < 5:
        print(">> SURPRISE: 60Hz stays 60Hz? (fs must be > 120Hz)")
    else:
        print(f">> UNEXPECTED: Peak at {peak_freq} Hz")

    # Plot
    plt.figure()
    plt.plot(freqs, psd)
    plt.title(f"PSD of Signal + 60Hz Sine (fs={fs})")
    plt.axvline(x=40, color='r', linestyle='--', label='40Hz (Alias)')
    plt.axvline(x=60, color='g', linestyle='--', label='60Hz (Target)')
    plt.legend()
    plt.savefig("debug_psd.png")
    print("Saved debug_psd.png")

if __name__ == "__main__":
    main()
