
import hydra
from omegaconf import DictConfig
import logging
import sys
import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import importlib
import inspect
from sklearn.metrics import precision_recall_curve, auc

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import ECGDataset

log = logging.getLogger(__name__)

# --- Re-use Helper Functions ---

def forward_logits(model, x):
    """Normalize forward outputs to logits tensor."""
    out = model(x)
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
        return out[0]
    if isinstance(out, dict):
        for k in ["logits", "y_hat", "pred", "output"]:
            if k in out and isinstance(out[k], torch.Tensor):
                return out[k]
    raise RuntimeError(f"Cannot extract logits from model output type={type(out)}")

def evaluate_predictions(model, loader, device, return_probs=True):
    model.eval()
    all_targets, all_probs = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting", leave=False):
            batch = [b.to(device) for b in batch]
            x, y = batch[:2]
            logits = forward_logits(model, x)
            probs = torch.softmax(logits, dim=1)
            
            all_targets.append(y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_targets), np.concatenate(all_probs)

def _pick_method_class(module, run_name: str):
    """Pick the most likely wrapper class from a module."""
    run = run_name.lower()
    candidates = []
    for name, obj in vars(module).items():
        if not inspect.isclass(obj): continue
        if obj.__module__ != module.__name__: continue
        try:
            if not issubclass(obj, torch.nn.Module): continue
        except Exception: continue
        candidates.append(obj)

    if not candidates:
        raise RuntimeError(f"No torch.nn.Module classes found in {module.__name__}")

    def score(cls):
        n = cls.__name__.lower()
        s = 0
        if "dann" in run and "dann" in n: s += 5
        if "vrex" in run and "vrex" in n: s += 5
        if "irm" in run and "irm" in n: s += 5
        if ("v2" in run or "disentangled" in run) and ("disentangled" in n or "pid" in n or "v2" in n): s += 5
        if "erm" in run and "erm" in n: s += 5
        s -= (len(n) / 100.0)
        return s

    candidates = sorted(candidates, key=score, reverse=True)
    return candidates[0]

def build_method_from_run_name(run_name: str, backbone, num_classes: int):
    rn = run_name.lower()
    if "dann" in rn: mod = "dann"
    elif "vrex" in rn: mod = "vrex"
    elif "irm" in rn: mod = "irm"
    elif rn == "v2" or "disentangled" in rn or "pid" in rn: mod = "disentangled"
    else: mod = "erm"

    module = importlib.import_module(f"src.methods.{mod}")
    cls = _pick_method_class(module, run_name)
    return cls(backbone, num_classes)

def _get_ids_from_splits(splits: dict, split_name: str):
    ids = splits.get(split_name, [])
    if ids: return ids
    out = []
    for k, v in splits.items():
        if split_name in str(k).lower():
            out.extend(v)
    return out

@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def main(cfg: DictConfig):
    log.info("Running standalone PR Curve Analysis...")
    
    # Path setup
    res_dir = "outputs"
    out_dir = "results/figures"
    os.makedirs(out_dir, exist_ok=True)
    
    methods = ["erm", "dann", "vrex", "irm", "v2"]
    pretty_names = {"erm": "ERM", "dann": "DANN", "vrex": "V-REx", "irm": "IRM", "v2": "PID (Ours)"}
    
    # Data Loading - OOD Only (Chapman)
    # We want to see OOD PR curves
    split_name = cfg.get("eval", {}).get("split", "test")
    
    manifest_df = pd.read_csv(cfg.data.paths.manifest_path)
    with open(cfg.data.paths.split_path, "r") as f:
        splits = json.load(f)
    
    ids = _get_ids_from_splits(splits, split_name)
    df = manifest_df[manifest_df["unique_id"].isin(ids)]
    
    # Filter for OOD (Chapman)
    df = df[df["dataset_source"] == "chapman"]
    
    if len(df) == 0:
        log.error("No Chapman (OOD) data found in the current split! Check splits.json.")
        return

    # Filter for processed file
    import h5py
    with h5py.File(cfg.data.paths.processed_path, "r") as f:
        existing = set(f.keys())
    df = df[df["unique_id"].isin(existing)].reset_index(drop=True)
    
    log.info(f"OOD Set Size: {len(df)}")
    
    ds = ECGDataset(df, cfg.data.paths.processed_path)
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pr_data = {} # Key: (method, condition) -> (prec, rec, auc)

    for method_key in methods:
        for condition in ["Clean", "Poisoned"]:
            run_name = f"{method_key}_fixed" if condition == "Clean" else f"{method_key}_60hz"
            if method_key == "v2" and condition == "Clean": run_name = "v2"
            
            sub_folder = "baselines" if condition == "Clean" else "shortcuts"
            ckpt_path = os.path.join(res_dir, sub_folder, run_name, "best_model.pt")
            
            if not os.path.exists(ckpt_path):
                log.warning(f"Checkpoint not found: {ckpt_path}")
                continue
                
            log.info(f"Analyzing {method_key} ({condition})...")
            
            # Load
            backbone = hydra.utils.instantiate(cfg.model)
            try:
                model = build_method_from_run_name(run_name, backbone, cfg.model.num_classes)
                state_dict = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(state_dict)
                model.to(device)
            except Exception as e:
                log.error(f"Error loading {run_name}: {e}")
                continue
            
            # Predict
            y_true, y_prob = evaluate_predictions(model, loader, device)
            
            # Calculate PR (Abnormal vs Normal)
            if cfg.model.num_classes == 2:
                y_true_bin = y_true
                y_score_bin = y_prob[:, 1]
            else:
                # Class 0 = Normal, Rest = Abnormal
                y_true_bin = (y_true != 0).astype(int)
                y_score_bin = 1.0 - y_prob[:, 0]
                
            prec, rec, _ = precision_recall_curve(y_true_bin, y_score_bin)
            pr_auc = auc(rec, prec)
            
            pr_data[(method_key, condition)] = (prec, rec, pr_auc)
            log.info(f"-> {condition} {method_key} AUPRC: {pr_auc:.4f}")

    # Plot
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, cond in enumerate(["Clean", "Poisoned"]):
        ax = axes[i]
        found_any = False
        for m_key in methods:
            if (m_key, cond) not in pr_data: continue
            prec, rec, pr_auc = pr_data[(m_key, cond)]
            ax.plot(rec, prec, label=f"{pretty_names[m_key]} (AUC={pr_auc:.2f})")
            found_any = True
            
        ax.set_title(f"PR Curve ({cond}) - OOD Detection")
        ax.set_xlabel("Recall (Sensitivity)")
        if i == 0: ax.set_ylabel("Precision (PPV)")
        if found_any: ax.legend()
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pr_curves_standalone.png"))
    plt.close()
    
    log.info(f"Saved PR Curves to {out_dir}/pr_curves_standalone.png")

if __name__ == "__main__":
    main()
