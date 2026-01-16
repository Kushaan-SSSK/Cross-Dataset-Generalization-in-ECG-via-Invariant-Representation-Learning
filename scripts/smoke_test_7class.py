
import sys
import os
import torch
from torch.utils.data import DataLoader
import numpy as np

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import ECGDataset
from src.const import LABEL_COL, NUM_CLASSES, CLASS_NAMES

def main():
    print(f"Testing 7-Class Enforcement. Expecting NUM_CLASSES={NUM_CLASSES}")
    
    # Setup paths
    processed_path = os.path.join("data", "processed", "signals.h5")
    manifest_path = os.path.join("data", "manifests", "master_manifest.csv")
    
    if not os.path.exists(processed_path):
        print(f"Error: {processed_path} not found.")
        return

    import pandas as pd
    df = pd.read_csv(manifest_path)
    
    # Test PTB-XL (Current labels should be 0, 6)
    print("\n--- Testing PTB-XL (Should pass validation if subset of 0-6) ---")
    df_ptb = df[df['dataset_source'] == 'ptbxl']
    try:
        ds_ptb = ECGDataset(df_ptb, processed_path, task_label_col=LABEL_COL)
        loader = DataLoader(ds_ptb, batch_size=10, num_workers=0)
        batch = next(iter(loader))
        print("Batch loaded successfully.")
        print(f"Labels: {batch[1]}")
        
    except ValueError as e:
        print(f"Validation Error (Expected?): {e}")

    # Test Chapman (Current labels 0-6)
    print("\n--- Testing Chapman (Should pass) ---")
    df_chap = df[df['dataset_source'] == 'chapman']
    try:
        ds_chap = ECGDataset(df_chap, processed_path, task_label_col=LABEL_COL)
        loader = DataLoader(ds_chap, batch_size=10, num_workers=0)
        batch = next(iter(loader))
        print("Batch loaded successfully.")
        print(f"Labels: {batch[1]}")
    except ValueError as e:
        print(f"Validation Error: {e}")

    # Test Dummy Invalid
    print("\n--- Testing Invalid Labels (Should fail) ---")
    df_inv = df_ptb.copy().head(10)
    df_inv[LABEL_COL] = 99 # Invalid label
    try:
        ds_inv = ECGDataset(df_inv, processed_path, task_label_col=LABEL_COL)
        print("Error: Invalid dataset initialized without error!")
    except ValueError as e:
        print(f"Caught Expected Validation Error: {e}")

if __name__ == "__main__":
    main()
