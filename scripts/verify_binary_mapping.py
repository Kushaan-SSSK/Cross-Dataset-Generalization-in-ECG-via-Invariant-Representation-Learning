import sys
import os
import pandas as pd
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import ECGDataset

# Config
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "signals.h5")
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "data", "manifests", "master_manifest.csv")

# Test Cases
TEST_IDS = {
    'ptbxl_1': 0,        # PTB-XL Label 0 -> Binary 0
    'ptbxl_8': 1,        # PTB-XL Label 6 -> Binary 1
    'chapman_JS00008': 0, # Chapman Label 0 -> Binary 0
    'chapman_JS10656': 1  # Chapman Label 2 -> Binary 1
}

def verify():
    print("--- Verifying Binary Label Mapping ---")
    
    # Load full manifest
    df = pd.read_csv(MANIFEST_PATH)
    
    # Filter to test IDs
    df_test = df[df['unique_id'].isin(TEST_IDS.keys())].copy()
    
    # Sort to match iteration order if needed, or just dict lookup
    # Instantiate Dataset with binary_labels=True
    ds = ECGDataset(df_test, PROCESSED_PATH, binary_labels=True)
    
    success = True
    
    for i in range(len(ds)):
        row = ds.manifest_df.iloc[i]
        uid = row['unique_id']
        expected = TEST_IDS.get(uid)
        
        # Get Item
        _, label, _ = ds[i]
        label_val = label.item()
        
        status = "OK" if label_val == expected else "FAIL"
        if label_val != expected: success = False
        
        print(f"ID: {uid:<20} | Orig: {row['task_a_label']} | Expected: {expected} | Got: {label_val} | {status}")
        
    if success:
        print("\nSUCCESS: All mappings correct.")
        sys.exit(0)
    else:
        print("\nFAILURE: Mapping errors found.")
        sys.exit(1)

if __name__ == "__main__":
    verify()
