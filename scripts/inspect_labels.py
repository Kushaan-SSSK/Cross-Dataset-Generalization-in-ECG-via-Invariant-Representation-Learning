
import os
import sys
import pandas as pd
import json
import numpy as np
from collections import Counter

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def main():
    # Paths
    manifest_path = "data/processed/master_manifest.csv"
    if not os.path.exists(manifest_path):
        manifest_path = "data/manifests/master_manifest.csv"
        
    split_path = "data/processed/splits.json"
    if not os.path.exists(split_path):
        split_path = "data/manifests/splits.json"
        
    print(f"Loading manifest from {manifest_path}")
    df = pd.read_csv(manifest_path)
    
    with open(split_path, 'r') as f:
        splits = json.load(f)
        
    # Flatten splits to get IDs if needed, or just use df
    # Let's look at source-wise distribution
    
    label_col = 'task_a_label'
    
    output_file = "labels_report.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Loading manifest from {manifest_path}\n")
        
        sources = df['dataset_source'].unique()
        f.write(f"Sources found: {sources}\n")
        
        for source in sources:
            f.write(f"\n--- Source: {source} ---\n")
            sub_df = df[df['dataset_source'] == source]
            
            labels = sub_df[label_col].dropna()
            counts = Counter(labels)
            total = len(labels)
            
            f.write(f"Total samples: {total}\n")
            f.write("Label Distribution:\n")
            sorted_keys = sorted(counts.keys())
            for k in sorted_keys:
                v = counts[k]
                f.write(f"  Class {k}: {v} ({v/total*100:.2f}%)\n")
                
            # Binary stats (0 vs Rest)
            n_zeros = counts.get(0, 0)
            n_rest = total - n_zeros
            f.write(f"Binary (0 vs Rest) Baseline: {max(n_zeros, n_rest)/total:.4f}\n")
            
    print(f"Report written to {output_file}")

if __name__ == "__main__":
    main()
