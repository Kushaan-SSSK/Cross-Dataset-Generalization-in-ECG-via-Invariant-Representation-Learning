
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import os

from src.dataset import ECGDataset
from src.utils.metrics import calculate_metrics
from src.utils.perturbations import Perturbations

log = logging.getLogger(__name__)

@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def main(cfg: DictConfig):
    log.info("Starting Stress Testing...")
    
    # 1. Load Data (Validation Set Only)
    manifest_df = pd.read_csv(cfg.data.paths.manifest_path)
    with open(cfg.data.paths.split_path, 'r') as f:
        splits = json.load(f)
    
    val_ids = splits.get('val', [])
    if not val_ids:
        for k, v in splits.items():
            if 'val' in k: val_ids.extend(v)
            
    # Filter 12-lead only
    val_df = manifest_df[manifest_df['unique_id'].isin(val_ids)]
    valid_sources = ['ptbxl', 'chapman']
    val_df = val_df[val_df['dataset_source'].isin(valid_sources)]
    
    # HDF5 key check
    import h5py
    with h5py.File(cfg.data.paths.processed_path, 'r') as f:
        existing = set(f.keys())
    val_df = val_df[val_df['unique_id'].isin(existing)].reset_index(drop=True)
    
    val_ds = ECGDataset(val_df, cfg.data.paths.processed_path)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False)
    
    # 2. Load Model
    # We need to know WHICH model to load.
    # Typically this script is run pointing to a specific checkpoint OR we iterate.
    # For now, let's assume we load 'best_model.pt' from the current dir 
    # OR the user passes a path via 'checkpoint_path'.
    
    # Let's assume the user runs this inside the hydra output dir of a trained model
    # OR we use the cfg to instantiate the same model structure.
    model = hydra.utils.instantiate(cfg.model)
    
    # allow configOverride
    checkpoint_path = cfg.get("checkpoint_path", "best_model.pt")
    if os.path.exists(checkpoint_path):
        # We need to load state dict. 
        # Note: The 'method' wraps the model. The state dict might be 'method' or 'model'.
        # Our training saved 'method.state_dict()'.
        
        # Instantiate Method (Generic wrapper to load weights)
        # We can use ERM as a generic wrapper for inference
        from src.methods.erm import ERM
        method = ERM(model, cfg.model.num_classes)
        
        state_dict = torch.load(checkpoint_path)
        method.load_state_dict(state_dict, strict=False)
        log.info(f"Loaded checkpoint from {checkpoint_path}")
    else:
        log.warning(f"No checkpoint found at {checkpoint_path}. Using random weights (Sanity Check Only).")
        from src.methods.erm import ERM
        method = ERM(model, cfg.model.num_classes)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    method.to(device)
    method.eval()
    
    # 3. Define Stress Tests
    # (Perturbation Name, Level)
    tests = [
        ('clean', 0.0),
        ('noise', 0.05),
        ('noise', 0.10),
        ('noise', 0.20),
        ('wander', 0.1),
        ('wander', 0.3),
        ('dropout', 0.1), # 10% lead dropout
        ('dropout', 0.5)  # 50% lead dropout
    ]
    
    results = []
    
    for name, level in tests:
        log.info(f"Running Test: {name} (Level {level})")
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"{name}-{level}"):
                batch = [b.to(device) for b in batch]
                x, y = batch[:2]
                
                # Apply Perturbation
                if name != 'clean':
                    x = Perturbations.apply_perturbation(x, name, level)
                
                # Forward
                logits = method(x)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.append(preds.cpu())
                all_targets.append(y.cpu())
                
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        
        metrics = calculate_metrics(all_preds, all_targets, cfg.model.num_classes)
        metric_row = {
            'perturbation': name,
            'level': level,
            'acc': metrics['val_acc'],
            'f1': metrics['val_f1']
        }
        results.append(metric_row)
        log.info(f"Result: {metric_row}")
        
    # Save Results
    # Save Results
    df_res = pd.DataFrame(results)
    output_name = cfg.get("result_name", "stress_test_results.csv")
    df_res.to_csv(output_name, index=False)
    log.info(f"Saved {output_name}")

if __name__ == "__main__":
    main()
