
import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import logging
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import ECGDataset
# Reuse helpers from updated run_paper_experiments
from scripts.run_paper_experiments import _safe_load_state_dict_for_eval, build_method_from_run_name, forward_logits

logging.basicConfig(level=logging.ERROR) 
log = logging.getLogger(__name__)

def compute_leakage_metrics(model, loader, device):
    """Computes both Accuracy and AUC for domain leakage."""
    model.eval()
    all_feats, all_domains = [], []
    
    # We need to extract features. 
    # Logic: For ResNet1d, features are output of avgpool.
    # We attach a hook.
    features_batch = []
    def hook(module, input, output):
        features_batch.append(output.view(output.size(0), -1))

    handle = None
    # Find avgpool layer
    net = model.model if hasattr(model, 'model') else model
    if hasattr(net, 'avgpool'):
        handle = net.avgpool.register_forward_hook(hook)
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Leakage", leave=False):
            x = batch[0].to(device)
            d = batch[2] # Domain is 3rd element
            
            features_batch.clear()
            if handle:
                _ = model(x) # Trigger hook
                if features_batch:
                    all_feats.append(features_batch[0].cpu().numpy())
            else:
                # Fallback if no avgpool (shouldn't happen for our ResNet)
                pass
            
            all_domains.append(d.cpu().numpy())
            
    if handle: handle.remove()
    
    if not all_feats: return 0.0, 0.5

    X = np.concatenate(all_feats)
    y = np.concatenate(all_domains)
    
    # Train Probe
    clf = LogisticRegression(max_iter=200, solver='liblinear')
    clf.fit(X, y)
    acc = clf.score(X, y)
    
    # AUC
    if len(np.unique(y)) == 2:
        probs = clf.predict_proba(X)[:, 1]
        try:
            auc = roc_auc_score(y, probs)
        except:
            auc = 0.5
    else:
        auc = 0.5 # Should be binary for PTB vs Chapman
        
    return acc, auc

def compute_flip_rate(model, loader_clean, loader_poisoned, device):
    """Computes the rate at which predictions change between clean and poisoned versions."""
    model.eval()
    flips = 0
    total = 0
    
    with torch.no_grad():
        for b_c, b_p in zip(loader_clean, loader_poisoned):
            x_c = b_c[0].to(device)
            x_p = b_p[0].to(device)
            
            logits_c = forward_logits(model, x_c)
            logits_p = forward_logits(model, x_p)
            
            pred_c = torch.argmax(logits_c, dim=1)
            pred_p = torch.argmax(logits_p, dim=1)
            
            flips += (pred_c != pred_p).sum().item()
            total += x_c.size(0)
            
    return flips / total if total > 0 else 0.0

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    processed_path = os.path.join(base_dir, "data", "processed", "signals.h5") 
    manifest_path = os.path.join(base_dir, "data", "manifests", "master_manifest.csv")
    
    df = pd.read_csv(manifest_path)
    
    # 1. Leakage Dataset (Mixed)
    # 500 from PTB, 500 from Chapman
    df_ptb = df[df['dataset_source'] == 'ptbxl'].head(500)
    df_chap = df[df['dataset_source'] == 'chapman'].head(500)
    df_leak = pd.concat([df_ptb, df_chap]).reset_index(drop=True)
    
    ds_leak = ECGDataset(df_leak, processed_path, task_label_col="task_a_label")
    l_leak = DataLoader(ds_leak, batch_size=64, shuffle=False)
    
    # 2. Flip Rate Dataset (PTB-XL Only)
    # Using 'test' split logic effectively (just first 500)
    # Need TWO datasets: Clean (no shortcut) and Poisoned (force shortcut)
    df_flip = df[df['dataset_source'] == 'ptbxl'].head(500).reset_index(drop=True)
    
    ds_cln = ECGDataset(df_flip, processed_path, task_label_col="task_a_label", shortcut_cfg=None, split='test')
    sc_cfg = OmegaConf.create({"use_shortcut": True, "freq": 60, "amplitude": 1.0, "correlation": 1.0})
    ds_pois = ECGDataset(df_flip, processed_path, task_label_col="task_a_label", shortcut_cfg=sc_cfg, split='train') # split=train enables injection
    
    l_cln = DataLoader(ds_cln, batch_size=64, shuffle=False)
    l_pois = DataLoader(ds_pois, batch_size=64, shuffle=False)
    
    print("Method,LeakAcc,LeakAUC,FlipRate")
    
    methods = ["erm", "dann", "vrex", "irm"]
    
    for m in methods:
        run_name = f"{m}_60hz"
        ckpt_path = os.path.join(base_dir, "outputs", "shortcuts", run_name, "best_model.pt")
        
        if not os.path.exists(ckpt_path):
            print(f"{m.upper()},N/A,N/A,N/A")
            continue
            
        # Load
        sd = torch.load(ckpt_path, map_location=device)
        
        # Infer classes
        num_c = 5
        if 'classifier.weight' in sd: num_c = sd['classifier.weight'].shape[0]
        elif 'model.fc.weight' in sd: num_c = sd['model.fc.weight'].shape[0]
        elif 'task_classifier.weight' in sd: num_c = sd['task_classifier.weight'].shape[0]
        elif 'classifier.bias' in sd: num_c = sd['classifier.bias'].shape[0]
        
        from src.models.resnet1d import ResNet1d
        backbone = ResNet1d(input_channels=12, num_classes=num_c)
        model = build_method_from_run_name(run_name, backbone, num_c)
        model = _safe_load_state_dict_for_eval(model, sd, strict=False)
        model.to(device)
        
        # Leakage
        acc, auc = compute_leakage_metrics(model, l_leak, device)
        
        # Flip Rate (Only relevant for SAST vulnerability check, primarily DANN)
        flip_rate = compute_flip_rate(model, l_cln, l_pois, device)
        
        print(f"{m.upper()},{acc:.4f},{auc:.4f},{flip_rate:.4f}")

if __name__ == "__main__":
    main()
