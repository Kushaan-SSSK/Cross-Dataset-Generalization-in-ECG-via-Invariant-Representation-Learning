
import hydra
from omegaconf import DictConfig
import logging
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import logging
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import json
import os
import importlib
import inspect
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc, log_loss
from sklearn.linear_model import LogisticRegression
from scipy.fft import fft

from src.dataset import ECGDataset
from omegaconf import OmegaConf
import re
logger = logging.getLogger(__name__)

# --- checkpoint key remapping helper (auto-added) ---
def _remap_state_dict_for_erm(state_dict):
    """Map DANN-style keys -> ERM expected keys.
    task_classifier.* -> model.fc.* ; drop domain_* keys
    """
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
    """Infer ResNet1d (planes, layers) by reading shapes + block indices from state_dict keys."""
    planes = [None, None, None, None]
    # infer planes from model.layer{n}.0.conv1.weight (out_channels)
    for li in range(1, 5):
        k = f"model.layer{li}.0.conv1.weight"
        if k in sd:
            planes[li-1] = int(sd[k].shape[0])
    # fallback for stage-1 if conv1 missing
    if planes[0] is None and 'model.conv1.weight' in sd:
        planes[0] = int(sd['model.conv1.weight'].shape[0])
    if any(p is None for p in planes):
        return None

    # infer layer counts by max block index in keys model.layerX.{idx}.
    layer_counts = [0, 0, 0, 0]
    rx = re.compile(r'^model\.layer([1-4])\.(\d+)\.')
    for k in sd.keys():
        m = rx.match(k)
        if not m:
            continue
        li = int(m.group(1)) - 1
        bi = int(m.group(2))
        layer_counts[li] = max(layer_counts[li], bi + 1)
    if any(c == 0 for c in layer_counts):
        return None
    return {'planes': planes, 'layers': layer_counts}

def _rebuild_resnet1d_from_sd(sd, default_model=None):
    """Rebuild src.models.resnet1d.ResNet1d to match checkpoint backbone widths."""
    arch = _infer_resnet1d_arch_from_sd(sd)
    if arch is None:
        raise RuntimeError('Could not infer ResNet1d arch from state_dict')
    from src.models.resnet1d import ResNet1d

    # num_classes from fc weight if present
    num_classes = None
    if 'model.fc.weight' in sd:
        num_classes = int(sd['model.fc.weight'].shape[0])
    elif 'task_classifier.weight' in sd:
        num_classes = int(sd['task_classifier.weight'].shape[0])
    else:
        # fallback
        num_classes = getattr(default_model, 'num_classes', None) or 7

    input_channels = getattr(default_model, 'input_channels', None) or 12
    m = ResNet1d(input_channels=input_channels, num_classes=num_classes, layers=arch['layers'], planes=arch['planes'])
    # stash for logging if class doesn't store
    m._inferred_layers = arch['layers']
    m._inferred_planes = arch['planes']
    return m

def _safe_load_state_dict_for_eval(model, state_dict, strict=True):
    """Robust load for eval.
    Handles:
      - DANN-style task_classifier/domain_* checkpoints
      - FC size mismatches by rebuilding ResNet1d from checkpoint backbone shapes
    Returns: (possibly rebuilt) model
    """
    log = logging.getLogger(__name__)
    sd = state_dict

    # If it's DANN-style, keep original sd around and also a remapped version
    has_task = isinstance(sd, dict) and any(k.startswith('task_classifier.') for k in sd.keys())
    remapped = _remap_state_dict_for_erm(sd) if has_task else sd

    try:
        model.load_state_dict(remapped, strict=strict)
        return model
    except RuntimeError as e:
        msg = str(e)
        # If we hit an FC mismatch, rebuild a matching ResNet1d from the checkpoint backbone and retry.
        if 'size mismatch for model.fc.weight' in msg or 'size mismatch for model.fc.bias' in msg:
            log.warning('FC size mismatch detected; attempting to rebuild ResNet1d to match checkpoint backbone')
            new_model = _rebuild_resnet1d_from_sd(remapped, default_model=model)
            log.warning('Rebuilt model with inferred planes=%s layers=%s', getattr(new_model, '_inferred_planes', None), getattr(new_model, '_inferred_layers', None))
            new_model.load_state_dict(remapped, strict=False)
            return new_model

        # Otherwise, if it is a DANN-style checkpoint, do the best-effort remap load
        if has_task:
            log.warning('Remapping task_classifier -> model.fc and dropping domain_* keys for ERM load')
            model.load_state_dict(remapped, strict=False)
            return model
        raise
# --- end checkpoint key remapping helper ---





log = logging.getLogger(__name__)

# --- Helper Functions ---

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

def forward_features(model, x):
    """
    Extract features from the backbone using a forward hook.
    This works whether the method returns logits or features, 
    and handles the different ways model.fc might be configured.
    """
    features = []
    
    def hook(module, input, output):
        # output of avgpool is (B, C, 1) or (B, C) depending on implementation details
        # ResNet1d avgpool is (B, C, 1).
        if output.dim() == 3:
            features.append(output.view(output.size(0), -1))
        else:
            features.append(output)

    # Find the backbone
    # Most methods store it in self.model
    # If model is the backbone itself (passed directly?), handle that.
    backbone = getattr(model, 'model', model)
    
    # Check for avgpool
    if hasattr(backbone, 'avgpool'):
        handle = backbone.avgpool.register_forward_hook(hook)
    else:
        # Fallback: Maybe it's not ResNet1d?
        # If we can't find avgpool, try just calling the model (maybe it returns features?)
        # But this is risky. Let's return logits as features if we fail? No.
        # Let's try wrapping the whole model.
        raise RuntimeError(f"Could not find avgpool in {type(backbone)}")
        
    try:
        _ = model(x)
    finally:
        handle.remove()
        
    if len(features) > 0:
        return features[0]
    raise RuntimeError("Hook did not capture features.")

def evaluate_model(model, loader, device, return_probs=False):
    model.eval()
    all_preds, all_targets, all_domains, all_probs = [], [], [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch = [b.to(device) for b in batch]
            x, y, d = batch[:3]
            logits = forward_logits(model, x)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_domains.append(d.cpu().numpy())
            
            if return_probs:
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    domains = np.concatenate(all_domains)
    
    if return_probs:
        probs = np.concatenate(all_probs)
        return preds, targets, domains, probs
    return preds, targets, domains

def _get_ids_from_splits(splits: dict, split_name: str):
    ids = splits.get(split_name, [])
    if ids:
        return ids
    out = []
    for k, v in splits.items():
        if split_name in str(k).lower():
            out.extend(v)
    return out

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

# --- Analysis Functions ---

def analyze_calibration(targets, probs, n_bins=10):
    """
    Compute ECE and Reliability Diagram data.
    Assumes binary classification for simplicity (probs is (N, 2)) or take max prob.
    """
    # Use max probability for confidence
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    
    accuracies = predictions == targets
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    ece = 0.0
    bin_stats = []
    
    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            bin_stats.append((avg_confidence_in_bin, accuracy_in_bin, prop_in_bin))
        else:
            bin_stats.append((0, 0, 0))
            
    return ece, bin_stats

def compute_dataset_leakage(model, loader, device):
    """
    Train a logistic regression probe on frozen features to predict Domain ID.
    Returns accuracy.
    """
    model.eval()
    all_feats, all_domains = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Leakage Probe", leave=False):
            x, _, d = [b.to(device) for b in batch[:3]]
            
            # Extract features
            feats = forward_features(model, x) # (B, F)
            if len(feats.shape) > 2:
                feats = feats.mean(dim=-1) # Global average pooling if (B, C, L)
            
            all_feats.append(feats.cpu().numpy())
            all_domains.append(d.cpu().numpy())
            
    X = np.concatenate(all_feats)
    y = np.concatenate(all_domains)
    
    # Train simple classifier
    clf = LogisticRegression(max_iter=200, solver='liblinear')
    clf.fit(X, y)
    acc = clf.score(X, y)
    
    # Calculate AUC (handle binary or multiclass)
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) == 2:
        y_probs = clf.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_probs)
    else:
        y_probs = clf.predict_proba(X)
        auc = roc_auc_score(y, y_probs, multi_class='ovr')

    log.info(f"Leakage: Acc={acc:.4f}, AUC={auc:.4f}")
    return acc # Return acc for compatibility, or tuple if callers update

def compute_frequency_attribution(model, loader, device, target_band=(58, 62), fs=100):
    """
    Compute Attribution Energy in target frequency band.
    Using Input Gradients (Saliency).
    """
    model.eval()
    attributions = []
    
    # Analyze a subset to save time
    limit = 200
    count = 0
    
    for batch in tqdm(loader, desc="Freq Attribution", leave=False):
        x, y = batch[0].to(device), batch[1].to(device)
        x.requires_grad = True
        
        logits = forward_logits(model, x)
        score = logits.gather(1, y.view(-1, 1)).squeeze()
        score.sum().backward()
        
        # Saliency: Abs gradients
        saliency = x.grad.abs().cpu().numpy() # (B, C, L)
        
        # Take Lead I (index 0)
        saliency_lead1 = saliency[:, 0, :] # (B, L)
        
        # FFT
        n = saliency_lead1.shape[1]
        freqs = np.fft.rfftfreq(n, d=1/fs)
        fft_vals = np.abs(np.fft.rfft(saliency_lead1, axis=1)) # (B, n/2+1)
        
        # Sum energy in band
        mask = (freqs >= target_band[0]) & (freqs <= target_band[1])
        band_energy = fft_vals[:, mask].sum(axis=1)
        total_energy = fft_vals.sum(axis=1)
        
        attributions.append(band_energy / (total_energy + 1e-8)) # Normalized
        
        count += len(x)
        if count >= limit:
            break
            
    return np.mean(np.concatenate(attributions))

# --- Main Script ---

@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def main(cfg: DictConfig):
    # Resolve config nesting: supports both `data.paths.*` and `data.data.paths.*`
    paths = OmegaConf.select(cfg, "data.paths") or OmegaConf.select(cfg, "data.data.paths")
    if paths is None:
        raise KeyError("Could not find paths at data.paths or data.data.paths")


    log.info("Generating Comprehensive Results...")

    # Configuration for results
    # User confirmed checkpoints are in outputs/
    res_dir = "outputs" 
    
    methods = ["erm", "dann", "vrex", "irm"]
    # Mapping for nice names
    pretty_names = {
        "erm": "ERM", 
        "dann": "DANN", 
        "vrex": "V-REx", 
        "irm": "IRM"
    }
    # Data Setup
    split_name = cfg.get("eval", {}).get("split", "test") # Default to test for final results
    log.info(f"Eval split: {split_name}")

    manifest_df = pd.read_csv(paths.manifest_path)
    with open(paths.split_path, "r") as f:
        splits = json.load(f)
    
    ids = _get_ids_from_splits(splits, split_name)
    df = manifest_df[manifest_df["unique_id"].isin(ids)]
    df = df[df["dataset_source"].isin(["ptbxl", "chapman"])]
    
    import h5py
    with h5py.File(paths.processed_path, "r") as f:
        existing = set(f.keys())
    df = df[df["unique_id"].isin(existing)].reset_index(drop=True)
    
    ds = ECGDataset(df, paths.processed_path)
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = "results/figures"
    os.makedirs(out_dir, exist_ok=True)

    # Domain mapping needed specifically for OOD check
    # In dataset.py: 0=PTB-XL (In-Domain), 1=Chapman (OOD)
    domain_map = {0: "In-Domain", 1: "OOD"}

    all_metrics = []
    calibration_data = {}
    pr_data = {}
    dfs_cm = []
    
    for method_key in methods:
        for condition in ["Clean", "Poisoned"]:
            run_name = f"{method_key}_fixed" if condition == "Clean" else f"{method_key}_60hz"
            if method_key == "v2" and condition == "Clean": run_name = "v2" # Exception based on dir structure
            
            # Determine path
            # Look in results/baselines for Clean, results/shortcuts for Poisoned
            sub_folder = "baselines" if condition == "Clean" else "shortcuts"
            ckpt_path = os.path.join(res_dir, sub_folder, run_name, "best_model.pt")
            
            if not os.path.exists(ckpt_path):
                log.warning(f"Checkpoint not found: {ckpt_path}. Skipping.")
                continue

            log.info(f"Processing {method_key} ({condition})...")
            
            # Load Model
            backbone = hydra.utils.instantiate(cfg.model)
            try:
                model = build_method_from_run_name(run_name, backbone, cfg.model.num_classes)
                state_dict = torch.load(ckpt_path, map_location=device)
                model = _safe_load_state_dict_for_eval(model, state_dict, logger)
                model.to(device)
            except Exception as e:
                log.error(f"Failed to load {run_name}: {e}")
                continue

            # 1. Standard Eval
            preds, targets, domains, probs = evaluate_model(model, loader, device, return_probs=True)
            
            # Metrics by Domain
            for d_idx, d_name in domain_map.items():
                mask = (domains == d_idx)
                if mask.sum() == 0: continue
                
                y_true = targets[mask]
                y_pred = preds[mask]
                y_prob = probs[mask]
                
                f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
                acc = accuracy_score(y_true, y_pred)
                try:
                    auroc = roc_auc_score(y_true, y_prob[:, 1]) if cfg.model.num_classes == 2 else 0
                except:
                    auroc = 0
                
                all_metrics.append({
                    "Method": pretty_names[method_key],
                    "Condition": condition,
                    "Domain": d_name,
                    "F1": f1,
                    "FullMethod": f"{method_key}_{condition}",
                    "AUROC": auroc
                })
                
                # Store OOD data for specific plots
                if d_name == "OOD":
                    # Calibration
                    ece, bin_stats = analyze_calibration(y_true, y_prob)
                    calibration_data[(method_key, condition)] = (ece, bin_stats)
                    all_metrics[-1]["ECE"] = ece
                    
                    # PR Curve (Binary or One-vs-Rest for Abnormal)
                    if cfg.model.num_classes == 2:
                        y_true_bin = y_true
                        y_score_bin = y_prob[:, 1]
                    else:
                        # Assume Class 0 is Normal, others Abnormal (Superdiagnostic)
                        y_true_bin = (y_true != 0).astype(int)
                        y_score_bin = 1.0 - y_prob[:, 0]

                    prec, rec, _ = precision_recall_curve(y_true_bin, y_score_bin)
                    pr_auc = auc(rec, prec)
                    pr_data[(method_key, condition)] = (prec, rec, pr_auc)
                    all_metrics[-1]["AUPRC"] = pr_auc
                    
                    # Confusion Matrix Data (Store for PID vs ERM)
                    if method_key in ["erm", "v2"]:
                        dfs_cm.append({
                             "method": method_key,
                             "condition": condition,
                             "y_true": y_true,
                             "y_pred": y_pred
                        })

            # 2. Leakage Analysis
            leak_acc = compute_dataset_leakage(model, loader, device)
            all_metrics.append({
                "Method": pretty_names[method_key],
                "Condition": condition,
                "Domain": "Leakage",
                "Accuracy": leak_acc,
                "FullMethod": f"{method_key}_{condition}"
            })
            
            # 3. Frequency Attribution (Poisoned Checks)
            # Use data sampling rate (default 250 in later runs, 100 in early)
            fs = cfg.data.get("sampling_rate", 100)
            freq_score = compute_frequency_attribution(model, loader, device, fs=fs)
            all_metrics.append({
                "Method": pretty_names[method_key],
                "Condition": condition,
                "Domain": "FreqAttribution",
                "Score": freq_score, 
                "FullMethod": f"{method_key}_{condition}"
            })

    # Save raw metrics
    df_res = pd.DataFrame(all_metrics)
    df_res.to_csv(os.path.join(out_dir, "all_results.csv"), index=False)
    
    # --- PLOTTING ---
    sns.set_theme(style="whitegrid")
    
    # 1. Performance Comparison (F1)
    plt.figure(figsize=(10, 6))
    if df_res is None or len(df_res) == 0:
        logging.getLogger(__name__).error("No results were generated (all model loads likely failed). Aborting figure generation.")
        return
    if "Domain" not in df_res.columns:
        logging.getLogger(__name__).error("Expected column 'Domain' not found. Columns: %s", list(df_res.columns))
        return

    subset = df_res[df_res["Domain"].isin(["In-Domain", "OOD"])]
    sns.barplot(data=subset, x="Method", y="F1", hue="Condition", ci=None) # ci=None as we have single run results here
    plt.title("Cross-Dataset Performance (Clean vs Poisoned)")
    plt.ylabel("Macro F1")
    plt.savefig(os.path.join(out_dir, "performance_comparison.png"))
    plt.close()
    
    # 2. Delta OOD Drop
    # Pivot to calculate drops
    drops = []
    for m in pretty_names.values():
        for c in ["Clean", "Poisoned"]:
            row_in = subset[(subset["Method"] == m) & (subset["Condition"] == c) & (subset["Domain"] == "In-Domain")]
            row_ood = subset[(subset["Method"] == m) & (subset["Condition"] == c) & (subset["Domain"] == "OOD")]
            if not row_in.empty and not row_ood.empty:
                drop = row_in.iloc[0]["F1"] - row_ood.iloc[0]["F1"]
                drops.append({"Method": m, "Condition": c, "Delta OOD": drop})
    
    plt.figure(figsize=(8, 5))
    sns.barplot(data=pd.DataFrame(drops), x="Method", y="Delta OOD", hue="Condition")
    plt.title("Performance Drop (In-Domain - OOD)")
    plt.ylabel("Δ F1")
    plt.savefig(os.path.join(out_dir, "ood_drop.png"))
    plt.close()
    
    # 3. Calibration
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for i, cond in enumerate(["Clean", "Poisoned"]):
        ax = axes[i]
        for m_key in methods:
            if (m_key, cond) not in calibration_data: continue
            ece, stats = calibration_data[(m_key, cond)]
            confs = [s[0] for s in stats]
            accs = [s[1] for s in stats]
            ax.plot(confs, accs, marker='o', label=f"{pretty_names[m_key]} (ECE={ece:.2f})")
        ax.plot([0,1], [0,1], 'k--', alpha=0.5)
        ax.set_title(f"{cond} Models (OOD)")
        ax.set_xlabel("Confidence")
        if i == 0: ax.set_ylabel("Accuracy")
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "calibration.png"))
    plt.close()
    
    # 4. PR Curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, cond in enumerate(["Clean", "Poisoned"]):
        ax = axes[i]
        for m_key in methods:
            if (m_key, cond) not in pr_data: continue
            prec, rec, pr_auc = pr_data[(m_key, cond)]
            ax.plot(rec, prec, label=f"{pretty_names[m_key]} (AUC={pr_auc:.2f})")
        ax.set_title(f"PR Curve ({cond}) - OOD")
        ax.set_xlabel("Recall")
        if i == 0: ax.set_ylabel("Precision")
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pr_curves.png"))
    plt.close()
    
    # 5. Dataset Leakage
    plt.figure(figsize=(8, 5))
    leak_df = df_res[df_res["Domain"] == "Leakage"]
    sns.barplot(data=leak_df, x="Method", y="Accuracy", hue="Condition")
    plt.axhline(0.5, color='r', linestyle='--', label="Random Guess")
    plt.title("Domain Identity Leakage (Probe Accuracy)")
    plt.ylabel("Probe Accuracy")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "leakage.png"))
    plt.close()
    
    # 6. Frequency Attribution
    plt.figure(figsize=(8, 5))
    freq_df = df_res[df_res["Domain"] == "FreqAttribution"]
    sns.barplot(data=freq_df, x="Method", y="Score", hue="Condition")
    plt.title("Attribution to 60Hz Band")
    plt.ylabel("Relative Energy")
    plt.savefig(os.path.join(out_dir, "freq_attribution.png"))
    plt.close()

    # 7. Confusion Matrix Diff (PID - ERM for Poisoned)
    # Find ERM Poisoned and PID Poisoned
    erm_p = next((x for x in dfs_cm if x["method"] == "erm" and x["condition"] == "Poisoned"), None)
    pid_p = next((x for x in dfs_cm if x["method"] == "v2" and x["condition"] == "Poisoned"), None)
    
    if erm_p and pid_p:
        cm_erm = confusion_matrix(erm_p["y_true"], erm_p["y_pred"], normalize='true')
        cm_pid = confusion_matrix(pid_p["y_true"], pid_p["y_pred"], normalize='true')
        
        diff = cm_pid - cm_erm
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(diff, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
        plt.title("Confusion Matrix Difference (PID - ERM) [Poisoned]")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.savefig(os.path.join(out_dir, "cm_diff_poisoned.png"))
        plt.close()

    log.info(f"Done! Results saved to {out_dir}")

if __name__ == "__main__":
    main()
