
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np
import json
from tqdm import tqdm

from src.dataset import ECGDataset
from src.utils.metrics import calculate_metrics

# Dynamically import models and methods based on config (or simple factory)
from src.models.resnet1d import ResNet1d
from src.methods.erm import ERM

# Setup logging
log = logging.getLogger(__name__)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def main(cfg: DictConfig):
    log.info(f"Training Config: \n{OmegaConf.to_yaml(cfg)}")
    
    seed_everything(cfg.seed)
    
    # 1. Load Data
    log.info("Loading Manifest and Splits...")
    # 1. Load Data
    log.info("Loading Manifest and Splits...")
    manifest_df = pd.read_csv(cfg.data.paths.manifest_path)
    with open(cfg.data.paths.split_path, 'r') as f:
        splits = json.load(f)
        
    # Determine split keys
    # Typically: 'train', 'val', 'test'
    # But splits.json structure might be dataset specific if we did cross-dataset.
    # Let's assume standard 'train', 'val' keys exist or we merge them.
    # Based on manifest builder, keys are likely 'train', 'val', 'test' lists of IDs.
    
    train_ids = splits.get('train', [])
    val_ids = splits.get('val', [])
    
    if not train_ids:
        # Fallback: check keys like 'ptbxl_train', etc and merge
        train_ids = []
        val_ids = []
        for k, v in splits.items():
            if 'train' in k:
                train_ids.extend(v)
            elif 'val' in k:
                val_ids.extend(v)
    
    log.info(f"Train Size: {len(train_ids)}, Val Size: {len(val_ids)}")
    
    # Filter manifest
    train_df = manifest_df[manifest_df['unique_id'].isin(train_ids)]
    val_df = manifest_df[manifest_df['unique_id'].isin(val_ids)]

    # FILTER: Exclude MIT-BIH (2-lead, variable length) for Task A (12-lead)
    # This prevents RuntimeError: stack expects each tensor to be equal size
    valid_sources = cfg.data.get('train_sources', ['ptbxl', 'chapman'])
    # Convert OmegaConf list to python list if needed
    if not isinstance(valid_sources, list):
         valid_sources = list(valid_sources)

    log.info(f"Filtering for sources: {valid_sources}")
    train_df = train_df[train_df['dataset_source'].isin(valid_sources)]
    val_df = val_df[val_df['dataset_source'].isin(valid_sources)]
    
    log.info(f"Filtered Train Size: {len(train_df)}, Val Size: {len(val_df)}")
    
    # Dataset
    train_ds = ECGDataset(train_df, cfg.data.paths.processed_path, shortcut_cfg=cfg.data.shortcut, split='train')
    val_ds = ECGDataset(val_df, cfg.data.paths.processed_path, shortcut_cfg=cfg.data.shortcut, split='val')
    
    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # 2. Model & Method
    log.info("Instantiating Model...")
    # Instantiate model using Hydra target or manual
    model = hydra.utils.instantiate(cfg.model)
    
    # Instantiate Method
    log.info(f"Instantiating Method: {cfg.method._target_}")
    method = hydra.utils.instantiate(
        cfg.method, 
        model=model, 
        num_classes=cfg.model.num_classes
    )
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    method.to(device)
    
    # Optimizer
    optimizer = method.configure_optimizers(cfg.train.lr, cfg.train.weight_decay)
    
    # 3. Training Loop
    best_val_f1 = 0.0
    
    for epoch in range(cfg.train.epochs):
        method.train()
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        steps = 0
        
        # Train
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.train.epochs} [Train]")
        for batch in pbar:
            optimizer.zero_grad()
            
            batch = [b.to(device) for b in batch]
            x, y = batch[:2] # Unpack for logging or specific use, but pass full batch to method usually? 
            # Actually, method.training_step takes 'batch'.
            # ERM expects (x,y) or batch[:2]. 
            # DANN expects (x,y,d).
            # So passing 'batch' (tuple of tensors on device) is correct.
            
            out = method.training_step(batch, steps)
            loss = out['loss']
            
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item()
            train_acc_sum += out['log']['train_acc']
            steps += 1
            
            pbar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = train_loss_sum / steps
        avg_train_acc = train_acc_sum / steps
        
        # Validation
        method.eval()
        val_losses = []
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                batch = [b.to(device) for b in batch]
                
                # Pass full batch to method (it handles unpacking)
                out = method.validation_step(batch, 0)
                val_losses.append(out['loss'].item())
                all_preds.append(out['preds'])
                all_targets.append(out['targets'])
        
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        
        metrics = calculate_metrics(all_preds, all_targets, cfg.model.num_classes)
        avg_val_loss = np.mean(val_losses)
        
        log.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} | Val Loss: {avg_val_loss:.4f}, Val Acc: {metrics['val_acc']:.4f}, Val F1: {metrics['val_f1']:.4f}")
        
        # Save Best
        # Save Best
        if metrics['val_f1'] > best_val_f1 or epoch == 0:
            best_val_f1 = metrics['val_f1']
            
            if hasattr(cfg, 'save_path'):
                 save_dir = cfg.save_path
            else:
                try:
                    save_dir = HydraConfig.get().run.dir
                except Exception:
                    save_dir = os.getcwd()
                
            save_path = os.path.join(save_dir, "best_model.pt")
            log.info(f"DEBUG: Saving best_model.pt to {save_path}")
            torch.save(method.state_dict(), save_path)
            log.info(f"DEBUG: File exists after save? {os.path.exists(save_path)}")
            log.info(f"New Best Model Saved! (F1: {best_val_f1:.4f})")

    # Save Last
    if hasattr(cfg, 'save_path'):
         save_dir = cfg.save_path
    else:
        try:
            save_dir = HydraConfig.get().run.dir
        except Exception:
            save_dir = os.getcwd()
    torch.save(method.state_dict(), os.path.join(save_dir, "last_model.pt"))

if __name__ == "__main__":
    main()
