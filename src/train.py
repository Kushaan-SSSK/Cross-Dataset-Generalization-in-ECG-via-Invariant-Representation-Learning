
import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf import OmegaConf
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
from pathlib import Path

log = logging.getLogger(__name__)

def _select_path(cfg, *keys: str, must_exist: bool = False, desc: str = "path") -> str:
    """
    Resolve a path from OmegaConf safely (works with struct mode), with repo-relative fallback.
    Preference order:
      1) First existing config key among `keys`
      2) Standard repo fallbacks based on `desc` (manifest/split/processed)
    """
    from pathlib import Path as _Path
    from omegaconf import OmegaConf as _OmegaConf

    # repo root = src/train.py -> parents[1]
    _REPO = _Path(__file__).resolve().parents[1]

    # 1) Config keys
    for k in keys:
        v = _OmegaConf.select(cfg, k)
        if v is None:
            continue
        cand = _Path(str(v))
        if not cand.is_absolute():
            cand = (_REPO / cand).resolve()
        if (not must_exist) or cand.exists():
            return str(cand)

    # 2) Known fallbacks
    fallbacks = []
    if desc == "manifest_path":
        fallbacks = [
            _REPO / "data" / "manifests" / "master_manifest.csv",
            _REPO / "data" / "processed" / "master_manifest.csv",
        ]
    elif desc == "split_path":
        fallbacks = [
            _REPO / "data" / "manifests" / "splits.json",
            _REPO / "data" / "processed" / "splits.json",
        ]
    elif desc == "processed_path":
        fallbacks = [
            _REPO / "data" / "processed" / "signals.h5",
        ]

    for fb in fallbacks:
        if (not must_exist) or fb.exists():
            return str(fb.resolve())

    tried = ", ".join(keys)
    fb_str = ", ".join(str(x) for x in fallbacks) if fallbacks else ""
    raise FileNotFoundError(
        f"Could not resolve {desc}. Tried config keys: {tried}"
        + (f" and fallbacks: {fb_str}" if fb_str else "")
    )

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def main(cfg: DictConfig):
    # PATCH: ensure torch is imported early in main()
    # A later 'import torch' inside main makes torch a local name.
    # Import it here so any early uses (device=...) do not crash.
    import torch

# PATCH: ensure cfg.data.shortcut exists
    # Some configs do not define data.shortcut, but code may access it.
    # In struct mode, missing keys raise ConfigAttributeError. Make it safe.
    from omegaconf import OmegaConf, open_dict
    if OmegaConf.select(cfg, "data.shortcut", default=None) is None:
        with open_dict(cfg):
            if "data" not in cfg:
                cfg.data = {}
            cfg.data.shortcut = {
                "use_shortcut": False,
                "type": "none",
                "correlation": 0.0,
                "force": False,
                "split": "train",
            }

    log.info(f"Training Config: \n{OmegaConf.to_yaml(cfg)}")
    
    seed_everything(cfg.seed)
    
    # 1. Load Data
    log.info("Loading Manifest and Splits...")
    # 1. Load Data
    log.info("Loading Manifest and Splits...")
    from src.const import LABEL_COL, NUM_CLASSES, CLASS_NAMES
    
    manifest_df = pd.read_csv(_select_path(cfg, 'data.paths.manifest_path', 'data.manifest_path', must_exist=True, desc='manifest_path'))
    
    # --- Runtime Label Validation (Prompt 1) ---
    # Ensure label column exists and is strictly valid
    if LABEL_COL not in manifest_df.columns:
         # Check if we need to map old column to new column (Prompt 2 will do this properly)
         # For now, if LABEL_COL doesn't exist, we might fail or look for 'task_a_label'
         if 'task_a_label' in manifest_df.columns and LABEL_COL == 'task_a_label':
             pass # OK
         else:
             log.warning(f"Label column {LABEL_COL} not found. Available: {manifest_df.columns}. Prompt 2 will fix this.")
    
    with open(_select_path(cfg, 'data.paths.split_path', 'data.split_path', must_exist=True, desc='split_path'), 'r') as f:
        splits = json.load(f)
        
    train_ids = splits.get('train', [])
    val_ids = splits.get('val', [])
    
    if not train_ids:
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
    valid_sources = cfg.data.get('train_sources', ['ptbxl', 'chapman'])
    if not isinstance(valid_sources, list):
         valid_sources = list(valid_sources)

    log.info(f"Filtering for sources: {valid_sources}")
    train_df = train_df[train_df['dataset_source'].isin(valid_sources)]
    
    # --- FIXED: Always exclude MIT-BIH from validation (incompatible shape) ---
    # First, exclude MIT-BIH unconditionally (2-lead data breaks batching)
    val_df = val_df[val_df['dataset_source'] != 'mitbih']
    _val_df_no_mitbih = val_df.copy()
    
    # Then, try to filter to train_sources for in-domain validation
    val_df_filtered = val_df[val_df['dataset_source'].isin(valid_sources)]
    
    if len(val_df_filtered) > 0:
        # Have in-domain validation data
        val_df = val_df_filtered
    else:
        # No in-domain validation - use OOD validation (but still excluding MIT-BIH)
        log.warning("Val became empty after filtering by train_sources=%s; using OOD validation (excluding mitbih).", valid_sources)
        val_df = _val_df_no_mitbih

    log.info(f"Filtered Train Size: {len(train_df)}, Val Size: {len(val_df)}")
    
    # Dataset - Enforce LABEL_COL
    # Note: ECGDataset logic now validates labels are 0-6 or fail.
    train_ds = ECGDataset(train_df, _select_path(cfg, 'data.paths.processed_path', 'data.processed_path', must_exist=True, desc='processed_path'), task_label_col=LABEL_COL, shortcut_cfg=cfg.data.shortcut, split='train')
    
    # AUTO-FIX: restore OOD val_ids/val_df before building val_ds (scripts/autofix_restore_ood_val_before_valds.py)
    # If filtering by train_sources wiped validation (common in ptbxl2chapman), restore OOD val split here.
    try:
        _train_srcs = list(getattr(getattr(cfg, 'data', None), 'train_sources', []) or [])
    except Exception:
        _train_srcs = []
    
    # Restore val_ids if empty (so val_ds is not empty)
    try:
        if 'val_ids' in locals() and isinstance(val_ids, (list, tuple)) and len(val_ids) == 0 and _train_srcs:
            if 'splits' in locals() and isinstance(splits, dict):
                if 'val' in splits:
                    val_ids = list(splits['val'])
                elif 'valid' in splits:
                    val_ids = list(splits['valid'])
    except Exception:
        pass
    
    # Restore val_df if empty (so any val_df-based dataset path is not empty)
    try:
        if 'val_df' in locals() and hasattr(val_df, '__len__') and len(val_df) == 0 and _train_srcs:
            if '_af_val_df_unfiltered' in locals() and _af_val_df_unfiltered is not None:
                val_df = _af_val_df_unfiltered
    except Exception:
        pass
    
    try:
        if _train_srcs:
            log.warning("AUTO-FIX: keeping OOD validation even though train_sources=%s (val_ids=%s, val_df_len=%s)", _train_srcs, (len(val_ids) if 'val_ids' in locals() else None), (len(val_df) if 'val_df' in locals() else None))
    except Exception:
        pass

    # AUTO-FIX: rebuild val_df from manifest_df+val_ids if val_df empty (scripts/autofix_rebuild_valdf_from_valids.py)
    # Your code path can end up with val_ids non-empty but val_df empty after source filtering.
    # val_ds is built from val_df, so rebuild val_df from manifest_df+val_ids here.
    try:
        if ('val_df' in locals()) and ('val_ids' in locals()) and ('manifest_df' in locals()):
            if val_df is not None and hasattr(val_df, '__len__') and len(val_df) == 0 and isinstance(val_ids, (list, tuple)) and len(val_ids) > 0:
                _id_col = None
                # common id columns first
                for _cand in ('record_id','ecg_id','study_id','exam_id','id','idx','index','filename','file','path'):
                    if hasattr(manifest_df, 'columns') and _cand in list(getattr(manifest_df, 'columns', [])):
                        _id_col = _cand
                        break
                # if not found, detect a column that matches val_ids
                if _id_col is None and hasattr(manifest_df, 'columns'):
                    for _c in list(manifest_df.columns):
                        try:
                            if manifest_df[_c].isin(val_ids).any():
                                _id_col = _c
                                break
                        except Exception:
                            pass
                if _id_col is not None:
                    val_df = manifest_df[manifest_df[_id_col].isin(val_ids)]
                    try:
                        log.warning("AUTO-FIX: rebuilt val_df from manifest_df using id_col=%s -> val_df_len=%s", _id_col, len(val_df))
                    except Exception:
                        print(f"[WARN] AUTO-FIX: rebuilt val_df using id_col={_id_col} -> val_df_len={len(val_df)}")
    except Exception:
        pass

    val_ds = ECGDataset(val_df, _select_path(cfg, 'data.paths.processed_path', 'data.processed_path', must_exist=True, desc='processed_path'), task_label_col=LABEL_COL, shortcut_cfg=cfg.data.shortcut, split='val')

    # AUTO-FIX v4: build train_loader/val_loader from train_ds/val_ds (inserted by scripts/autofix_loaders_v4.py)
    # Build loaders from datasets that your code already created (train_ds / val_ds).
    def _af_get(_obj, _path, _default=None):
        try:
            cur = _obj
            for part in _path.split("."):
                cur = getattr(cur, part)
            return cur
        except Exception:
            return _default
    
    if 'train_loader' not in locals() or train_loader is None:
        _bs = _af_get(cfg, "data.batch_size", None) or _af_get(cfg, "train.batch_size", None) or 32
        _nw = _af_get(cfg, "data.num_workers", 0) or 0
        _pm = _af_get(cfg, "data.pin_memory", True)
        train_loader = DataLoader(train_ds, batch_size=int(_bs), shuffle=True, num_workers=int(_nw), pin_memory=bool(_pm))
    
    if 'val_loader' not in locals() and 'val_ds' in locals():
        _bs = _af_get(cfg, "data.batch_size", None) or _af_get(cfg, "train.batch_size", None) or 32
        _nw = _af_get(cfg, "data.num_workers", 0) or 0
        _pm = _af_get(cfg, "data.pin_memory", True)
        val_loader = DataLoader(val_ds, batch_size=int(_bs), shuffle=False, num_workers=int(_nw), pin_memory=bool(_pm))
    
    # 2. Model & Method
    log.info("Instantiating Model...")
    
    # FORCE NUM_CLASSES (Prompt 1: "enforce it programmatically")
    log.info(f"Enforcing NUM_CLASSES={NUM_CLASSES}")
    OmegaConf.set_struct(cfg, False)
    cfg.model.num_classes = NUM_CLASSES 
    
    # Instantiate model
    model = hydra.utils.instantiate(cfg.model)
    
    # --- Compute class weights for imbalanced data ---
    use_class_weights = OmegaConf.select(cfg, 'train.use_class_weights', default=True)
    class_weights = None
    if use_class_weights:
        from sklearn.utils.class_weight import compute_class_weight
        labels = train_df[LABEL_COL].values
        # Compute balanced class weights (inverse frequency)
        class_weights = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
        log.info(f"Using class weights: {class_weights.tolist()}")
    
    # Instantiate Method
    log.info(f"Instantiating Method: {cfg.method._target_}")
    method = hydra.utils.instantiate(
        cfg.method, 
        model=model, 
        num_classes=NUM_CLASSES,
        class_weights=class_weights
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
            
            # PATCH: normalize binary labels to 0/1
            
            # Some tasks use non-contiguous label IDs for binary classification (e.g., {0,6}).
            
            # CrossEntropyLoss requires labels in [0..C-1]. If C==2, map any nonzero -> 1.
            
            try:
            
                import torch
            
                if int(getattr(cfg.model, "num_classes", -1)) == 2:
            
                    _y = None
            
                    _key = None
            
                    if isinstance(batch, dict):
            
                        for _k in ("y", "label", "labels", "target", "targets"):
            
                            if _k in batch:
            
                                _y = batch[_k]
            
                                _key = _k
            
                                break
            
                    elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
            
                        _y = batch[1]
            
                    if _y is not None and torch.is_tensor(_y) and _y.numel() > 0:
            
                        if int(_y.max().item()) > 1:
            
                            _y2 = (_y > 0).long()
            
                            if isinstance(batch, dict) and _key is not None:
            
                                batch[_key] = _y2
            
                            elif isinstance(batch, tuple):
            
                                batch = (batch[0], _y2, *batch[2:])
            
                            elif isinstance(batch, list):
            
                                batch[1] = _y2
            
            except Exception:
            
                pass

            
            # PATCH: normalize binary labels before step calls
            # If running binary CE (C=2) but labels are encoded as {0,6} etc, remap nonzero -> 1.
            try:
                if int(getattr(getattr(cfg, "model", object()), "num_classes", -1)) == 2:
                    _y = None
                    _key = None
                    if isinstance(batch, dict):
                        for _k in ("y", "label", "labels", "target", "targets"):
                            if _k in batch:
                                _y = batch[_k]
                                _key = _k
                                break
                    elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
                        _y = batch[1]
                    if _y is not None and hasattr(_y, "numel") and _y.numel() > 0:
                        # torch tensor check without importing torch locally
                        _maxy = int(_y.max().item()) if hasattr(_y, "max") else None
                        if _maxy is not None and _maxy > 1:
                            _y2 = (_y > 0).long()
                            if isinstance(batch, dict) and _key is not None:
                                batch[_key] = _y2
                            elif isinstance(batch, tuple):
                                batch = (batch[0], _y2, *batch[2:])
                            elif isinstance(batch, list):
                                batch[1] = _y2
            except Exception:
                pass

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
                # PATCH: normalize binary labels before step calls
                # If running binary CE (C=2) but labels are encoded as {0,6} etc, remap nonzero -> 1.
                try:
                    if int(getattr(getattr(cfg, "model", object()), "num_classes", -1)) == 2:
                        _y = None
                        _key = None
                        if isinstance(batch, dict):
                            for _k in ("y", "label", "labels", "target", "targets"):
                                if _k in batch:
                                    _y = batch[_k]
                                    _key = _k
                                    break
                        elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
                            _y = batch[1]
                        if _y is not None and hasattr(_y, "numel") and _y.numel() > 0:
                            # torch tensor check without importing torch locally
                            _maxy = int(_y.max().item()) if hasattr(_y, "max") else None
                            if _maxy is not None and _maxy > 1:
                                _y2 = (_y > 0).long()
                                if isinstance(batch, dict) and _key is not None:
                                    batch[_key] = _y2
                                elif isinstance(batch, tuple):
                                    batch = (batch[0], _y2, *batch[2:])
                                elif isinstance(batch, list):
                                    batch[1] = _y2
                except Exception:
                    pass

                out = method.validation_step(batch, 0)
                val_losses.append(out['loss'].item())
                all_preds.append(out['preds'])
                all_targets.append(out['targets'])
        
        # AUTO-FIX empty-cat v1 (inserted by scripts/autofix_empty_cat_v1.py)
        # Prevent torch.cat() on empty collections (usually empty val_ds/val_loader for some splits).
        if len(all_preds) == 0:
            try:
                _vlen = len(val_ds) if 'val_ds' in locals() and val_ds is not None else None
            except Exception:
                _vlen = None
            try:
                _vbatches = len(val_loader) if 'val_loader' in locals() and val_loader is not None else None
            except Exception:
                _vbatches = None
            print(f"[WARN] No validation batches collected this epoch (val_ds_len={_vlen}, val_loader_len={_vbatches}). Skipping metric aggregation.")
            # Skip the rest of the validation aggregation for this epoch rather than crashing.
            # This assumes this code is inside the `for epoch in range(...)` loop (it is in your trace).
            continue

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
            # ensure save directory exists
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
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
