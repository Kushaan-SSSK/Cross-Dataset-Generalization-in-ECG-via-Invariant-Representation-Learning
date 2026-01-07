
import hydra
from omegaconf import DictConfig, OmegaConf
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
from scipy import signal

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import ECGDataset
import re

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --- Helper Functions ---

def _remap_state_dict_for_erm(state_dict):
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith('task_classifier.'):
            new_sd['model.fc.' + k.split('.', 1)[1]] = v
        elif k.startswith('domain_') or k.startswith('domain_adversary.') or k.startswith('domain_predictor.'):
            continue
        else:
            new_sd[k] = v
    return new_sd

def _infer_resnet1d_arch_from_sd(sd):
    planes = [None, None, None, None]
    for li in range(1, 5):
        k = f"model.layer{li}.0.conv1.weight"
        if k in sd:
            planes[li-1] = int(sd[k].shape[0])
    if planes[0] is None and 'model.conv1.weight' in sd:
        planes[0] = int(sd['model.conv1.weight'].shape[0])
    if any(x is None for x in planes): return None
    layer_counts = [0, 0, 0, 0]
    rx = re.compile(r'^model\.layer([1-4])\.(\d+)\.')
    for k in sd.keys():
        m = rx.match(k)
        if m:
            li = int(m.group(1)) - 1
            bi = int(m.group(2))
            layer_counts[li] = max(layer_counts[li], bi + 1)
    if any(c == 0 for c in layer_counts): return None
    return {'planes': planes, 'layers': layer_counts}

def _rebuild_resnet1d_from_sd(sd, default_model=None):
    arch = _infer_resnet1d_arch_from_sd(sd)
    if arch is None: raise RuntimeError('Could not infer ResNet1d arch')
    from src.models.resnet1d import ResNet1d
    if 'model.fc.weight' in sd: num_classes = int(sd['model.fc.weight'].shape[0])
    elif 'task_classifier.weight' in sd: num_classes = int(sd['task_classifier.weight'].shape[0])
    else: num_classes = getattr(default_model, 'num_classes', None) or 7
    input_channels = getattr(default_model, 'input_channels', None) or 12
    m = ResNet1d(input_channels=input_channels, num_classes=num_classes, layers=arch['layers'], planes=arch['planes'])
    m._inferred_layers = arch['layers']
    m._inferred_planes = arch['planes']
    return m

def _safe_load_state_dict_for_eval(model, state_dict, strict=True):
    sd = state_dict
    has_task = isinstance(sd, dict) and any(k.startswith('task_classifier.') for k in sd.keys())
    remapped = _remap_state_dict_for_erm(sd) if has_task else sd
    try:
        model.load_state_dict(remapped, strict=strict)
        return model
    except RuntimeError as e:
        msg = str(e)
        if 'size mismatch for model.fc.weight' in msg or 'size mismatch for model.fc.bias' in msg:
            new_model = _rebuild_resnet1d_from_sd(remapped, default_model=model)
            new_model.load_state_dict(remapped, strict=False)
            return new_model
        if has_task:
            model.load_state_dict(remapped, strict=False)
            return model
        raise

def _pick_method_class(module, run_name: str):
    # Determine class based on run_name and module content
    run = run_name.lower()
    candidates = []
    for name, obj in vars(module).items():
        if inspect.isclass(obj) and obj.__module__ == module.__name__:
            try:
                if issubclass(obj, torch.nn.Module): candidates.append(obj)
            except Exception: continue
    if not candidates: raise RuntimeError(f"No torch.nn.Module classes found in {module.__name__}")
    
    # Simple scoring to find best match
    def score(cls):
        n = cls.__name__.lower()
        s = 0
        if "dann" in run and "dann" in n: s += 5
        if "vrex" in run and "vrex" in n: s += 5
        if "erm" in run and "erm" in n: s += 5
        s -= (len(n) / 100.0)
        return s
    
    candidates = sorted(candidates, key=score, reverse=True)
    return candidates[0]

def build_method_from_run_name(run_name: str, backbone, num_classes: int):
    rn = run_name.lower()
    if "dann" in rn: mod = "dann"
    elif "vrex" in rn: mod = "vrex"
    else: mod = "erm"
    
    module = importlib.import_module(f"src.methods.{mod}")
    cls = _pick_method_class(module, run_name)
    return cls(backbone, num_classes)

def forward_logits(model, x):
    out = model(x)
    if isinstance(out, torch.Tensor): return out
    if isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor): return out[0]
    if isinstance(out, dict):
        for k in ["logits", "pred", "output"]:
            if k in out and isinstance(out[k], torch.Tensor): return out[k]
    raise RuntimeError(f"Cannot extract logits from type={type(out)}")

def get_ids_from_splits(splits: dict, split_name: str):
    ids = splits.get(split_name, [])
    if ids: return ids
    out = []
    for k, v in splits.items():
        if split_name in str(k).lower():
            out.extend(v)
    return out

# --- Experiment Logic ---

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            x, y = batch[0].to(device), batch[1].to(device)
            logits = forward_logits(model, x)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    
    from sklearn.metrics import f1_score
    f1 = f1_score(targets, preds, average="macro", zero_division=0)
    return f1

def main():
    # --- Configuration ---
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    manifest_path = os.path.join(base_dir, "data", "manifests", "master_manifest.csv")
    split_path = os.path.join(base_dir, "data", "manifests", "splits.json")
    processed_path = os.path.join(base_dir, "data", "processed", "signals.h5") 
    
    # Artifacts to test (Corresponding to planned experiments)
    # Checkpoints expected at: outputs/shortcuts/{method}_{artifact}
    artifacts = ['60hz'] # Start with 60Hz. Add 'bw', 'emg' if you train them.
    methods = ['erm', 'dann']
    
    log.info("Running EMBC Rescue Experiments...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Data manifests
    manifest_df = pd.read_csv(manifest_path)
    with open(split_path, "r") as f: splits = json.load(f)
    
    # 1. Prepare Datasets (Source: PTB-XL, Target: Chapman)
    # We need:
    #   - Source Test CLEAN (Control)
    #   - Source Test POISONED (Same-Task Stress)
    #   - Target Test CLEAN (Generalization)
    #   - Target Test POISONED (Cross-Task Stress)
    
    import h5py
    with h5py.File(processed_path, "r") as f: existing = set(f.keys())

    def get_loader(source_name, split_name, sc_cfg, batch_size=128):
        ids = get_ids_from_splits(splits, split_name)
        df = manifest_df[manifest_df["unique_id"].isin(ids)]
        if source_name:
            df = df[df["dataset_source"] == source_name]
        df = df[df["unique_id"].isin(existing)].reset_index(drop=True)
        # Assuming task_label_col is consistent or we handle mapping. 
        # For simplicity, using 'task_a_label' (Superdiagnostic / Rhythm)
        ds = ECGDataset(df, processed_path, task_label_col="task_a_label", shortcut_cfg=sc_cfg, split='test')
        return DataLoader(ds, batch_size=batch_size, shuffle=False)

    results = []

    for art in artifacts:
        log.info(f"--- Evaluating Artifact: {art.upper()} ---")
        
        # Shortcut Configs
        cfg_clean = None
        cfg_pois = OmegaConf.create({
            "use_shortcut": True, 
            "type": "mains" if art == "60hz" else art, # Map '60hz' -> 'mains'
            "freq": 60, "amplitude": 1.0, 
            "force": True # FORCE INJECTION
        })

        for m in methods:
            run_name = f"{m}_{art}"
            ckpt_path = os.path.join(base_dir, "outputs", "shortcuts", run_name, "best_model.pt")
            
            if not os.path.exists(ckpt_path):
                log.warning(f"Checkpoint not found: {ckpt_path}. Skipping.")
                continue
                
            log.info(f"Loading {m} from {run_name}...")
            sd = torch.load(ckpt_path, map_location=device)
            
            # Rebuild Model
            # Hack: infer num_classes
            if 'classifier.weight' in sd: nc = sd['classifier.weight'].shape[0]
            elif 'model.fc.weight' in sd: nc = sd['model.fc.weight'].shape[0]
            else: nc = 5 # Default
            
            from src.models.resnet1d import ResNet1d
            backbone = ResNet1d(input_channels=12, num_classes=nc)
            model = build_method_from_run_name(run_name, backbone, nc)
            model = _safe_load_state_dict_for_eval(model, sd)
            model.to(device)
            
            # --- EVALUATION LOOP ---
            
            # 1. Source (PTB-XL) - Same Task Control
            l_src_clean = get_loader('ptbxl', 'test', cfg_clean)
            l_src_pois = get_loader('ptbxl', 'test', cfg_pois)
            
            f1_src_c = evaluate(model, l_src_clean, device)
            f1_src_p = evaluate(model, l_src_pois, device)
            
            # 2. Target (Chapman) - Cross Task Generalization
            # 2. Target (Chapman) - Cross Task Generalization
            # Test Clean (Ablation: What if we remove shortcut?)
            l_tgt_clean = get_loader('chapman', 'test', cfg_clean)
            # Test Poisoned (Standard SAST)
            l_tgt_pois = get_loader('chapman', 'test', cfg_pois)
            
            # Test Notch Filtered (Augmentation: DANN + Notch)
            # We use Pois cfg but set apply_notch=True
            cfg_notch = OmegaConf.create({
                "use_shortcut": True, 
                "type": "mains" if art == "60hz" else art, 
                "freq": 60, "amplitude": 1.0, 
                "force": True,
                "apply_notch": True # NEW FLAG
            })
            l_tgt_notch = get_loader('chapman', 'test', cfg_notch)

            f1_tgt_c = evaluate(model, l_tgt_clean, device) # Ablation (Pois Model on Clean)
            f1_tgt_p = evaluate(model, l_tgt_pois, device)  # Standard (Pois Model on Pois)
            f1_tgt_n = evaluate(model, l_tgt_notch, device) # Augmentation (Pois Model on Notch)
            
            row = {
                "Method": m.upper(),
                "Artifact": art.upper(),
                "Tgt_Clean_Ablation": f1_tgt_c, 
                "Tgt_Pois_SAST": f1_tgt_p,
                "Tgt_Notch_Aug": f1_tgt_n,
                "Ablation_Gap": f1_tgt_c - f1_tgt_p,
                "Notch_Improvement": f1_tgt_n - f1_tgt_p
            }
            results.append(row)
            print(f"Result {m.upper()}: AblationGap={row['Ablation_Gap']:.4f}, NotchImp={row['Notch_Improvement']:.4f}")

    # Save
    if results:
        df_res = pd.DataFrame(results)
        out_path = os.path.join(base_dir, "results", "figures", "embc_rescue_results.csv")
        df_res.to_csv(out_path, index=False)
        print(f"\nSaved results to {out_path}")
        print(df_res)
    else:
        print("No results generated (checkpoints missing?)")

if __name__ == "__main__":
    main()
