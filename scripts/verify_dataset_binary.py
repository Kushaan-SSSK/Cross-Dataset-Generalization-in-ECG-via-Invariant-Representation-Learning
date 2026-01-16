
import sys
import os
import torch
from torch.utils.data import DataLoader
import numpy as np

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import ECGDataset

def main():
    # Setup paths
    processed_path = os.path.join("data", "processed", "signals.h5")
    manifest_path = os.path.join("data", "manifests", "master_manifest.csv")
    
    if not os.path.exists(processed_path):
        print(f"Error: {processed_path} not found.")
        return

    import pandas as pd
    df = pd.read_csv(manifest_path)
    
    # Test PTB-XL
    print("\n--- Testing PTB-XL Binary Mapping ---")
    df_ptb = df[df['dataset_source'] == 'ptbxl']
    ds_ptb = ECGDataset(df_ptb, processed_path, task_label_col='task_a_label', binary_labels=True)
    loader = DataLoader(ds_ptb, batch_size=1024, num_workers=0)
    
    all_y = []
    for _, y, _ in loader:
        all_y.append(y.numpy())
    all_y = np.concatenate(all_y)
    
    print(f"Unique Labels: {np.unique(all_y)}")
    counts = np.bincount(all_y)
    print(f"Distribution: {counts} (Ratio: {counts/len(all_y)})")
    
    # Test Chapman
    print("\n--- Testing Chapman Binary Mapping ---")
    df_chap = df[df['dataset_source'] == 'chapman']
    ds_chap = ECGDataset(df_chap, processed_path, task_label_col='task_a_label', binary_labels=True)
    loader = DataLoader(ds_chap, batch_size=1024, num_workers=0)
    
    all_y = []
    for _, y, _ in loader:
        all_y.append(y.numpy())
    all_y = np.concatenate(all_y)
    
    print(f"Unique Labels: {np.unique(all_y)}")
    counts = np.bincount(all_y)
    print(f"Distribution: {counts} (Ratio: {counts/len(all_y)})")
    
    # Check if this matches report
    # PTB: Class 0 ~44%, Class 1 ~56%
    # Chapman: Class 0 ~17%, Class 1 ~83%

if __name__ == "__main__":
    main()
