
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import os
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from src.dataset import ECGDataset
from src.methods.erm import ERM

log = logging.getLogger(__name__)

def evaluate_model(model, loader, device, num_classes):
    model.eval()
    all_preds = []
    all_targets = []
    all_domains = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            batch = [b.to(device) for b in batch]
            x, y, d = batch[:3]
            
            # Forward (using ERM wrapper which just calls model(x))
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_domains.append(d.cpu().numpy())
            
    return np.concatenate(all_preds), np.concatenate(all_targets), np.concatenate(all_domains)

@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def main(cfg: DictConfig):
    log.info("Generating Results...")
    
    # 1. Load Data
    manifest_df = pd.read_csv(cfg.data.paths.manifest_path)
    with open(cfg.data.paths.split_path, 'r') as f:
        splits = json.load(f)
    
    val_ids = splits.get('val', [])
    if not val_ids:
        for k, v in splits.items():
            if 'val' in k: val_ids.extend(v)
            
    val_df = manifest_df[manifest_df['unique_id'].isin(val_ids)]
    valid_sources = ['ptbxl', 'chapman']
    val_df = val_df[val_df['dataset_source'].isin(valid_sources)]
    
    import h5py
    with h5py.File(cfg.data.paths.processed_path, 'r') as f:
        existing = set(f.keys())
    val_df = val_df[val_df['unique_id'].isin(existing)].reset_index(drop=True)
    
    val_ds = ECGDataset(val_df, cfg.data.paths.processed_path)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False)
    
    # 2. Define Models to Evaluate
    # We assume standard output paths: outputs/baselines/erm, outputs/baselines/dann, etc.
    # Adjust base_dir as needed.
    base_dir = "outputs/baselines" 
    methods = ['erm', 'dann', 'vrex']
    
    results = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Backbone
    backbone = hydra.utils.instantiate(cfg.model)
    
    for method_name in methods:
        path = os.path.join(base_dir, method_name, "best_model.pt")
        if not os.path.exists(path):
            log.warning(f"Checkpoint not found for {method_name} at {path}. Skipping.")
            continue
            
        log.info(f"Evaluating {method_name}...")
        
        # Load Weights
        # We wrap in generic ERM to get standard forward() behavior
        model_wrapper = ERM(backbone, cfg.model.num_classes)
        state_dict = torch.load(path, map_location=device)
        model_wrapper.load_state_dict(state_dict)
        model_wrapper.to(device)
        
        preds, targets, domains = evaluate_model(model_wrapper, val_loader, device, cfg.model.num_classes)
        
        # Overall Metrics
        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average='macro', zero_division=0)
        results.append({'method': method_name, 'domain': 'Overall', 'acc': acc, 'f1': f1})
        
        # Per-Domain Metrics
        domain_map = {0: 'PTB-XL', 1: 'Chapman'}
        for d_idx, d_name in domain_map.items():
            mask = (domains == d_idx)
            if mask.sum() > 0:
                d_acc = accuracy_score(targets[mask], preds[mask])
                d_f1 = f1_score(targets[mask], preds[mask], average='macro', zero_division=0)
                results.append({'method': method_name, 'domain': d_name, 'acc': d_acc, 'f1': d_f1})
                
        # Confusion Matrix (Overall)
        cm = confusion_matrix(targets, preds, normalize='true')
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues')
        plt.title(f'Confusion Matrix: {method_name.upper()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        os.makedirs("results", exist_ok=True)
        plt.savefig(f"results/cm_{method_name}.png")
        plt.close()
        
    if not results:
        log.warning("No models evaluated.")
        return

    # 3. Save and Plot Table
    df_res = pd.DataFrame(results)
    df_res.to_csv("results/final_metrics.csv", index=False)
    print("\n=== Final Results ===")
    print(df_res)
    
    # Bar Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_res, x='method', y='f1', hue='domain')
    plt.title("Method Comparison by Domain")
    plt.ylabel("Macro F1 Score")
    plt.savefig("results/method_comparison.png")
    plt.close()
    
    log.info("Results generation complete. Check 'results/' folder.")

if __name__ == "__main__":
    main()
