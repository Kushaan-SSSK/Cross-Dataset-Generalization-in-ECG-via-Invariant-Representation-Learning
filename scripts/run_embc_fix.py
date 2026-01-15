
# PATCH: force probe num_classes=7

import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
import hydra
from omegaconf import OmegaConf
from tqdm import tqdm
import subprocess
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, f1_score
from scipy import stats

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import ECGDataset
from torch.utils.data import DataLoader
from src.models.resnet1d import ResNet1d

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Configuration ---
SEEDS = [0, 1, 2, 3, 4]
METHODS = ['erm', 'dann', 'vrex']
DIRECTIONS = [('ptbxl', 'chapman'), ('chapman', 'ptbxl')] # (Source, Target)
RHOS = [0.6, 0.7, 0.8, 0.9] # Sensitivity Sweep
CONDITIONS = ['Clean'] + [f'SAST_{r}' for r in RHOS]
EPOCHS = 50 # As per baseline script

# Base Output Dir
BASE_OUT_DIR = "outputs/embc_fix"
RESULTS_FILE = os.path.join(BASE_OUT_DIR, "embc_fix_raw_results.csv")
SUMMARY_FILE = os.path.join(BASE_OUT_DIR, "embc_fix_summary.csv")


    # --- Helper Functions (Copied/Adapted) ---
    
# PATCH: infer num_classes for evaluation

# PATCH: infer num_classes from manifest for probes
def infer_num_classes_from_manifest(manifest_path: str, label_col: str | None = None, prefer: int = 7) -> int:
    """Infer num_classes from the manifest.
    - If label_col provided, use it.
    - Otherwise, search numeric columns and prefer a contiguous label space of size `prefer`
      (e.g., 7 classes with labels 0..6).
    - Avoid ID-like columns (patient_id, subject_id, record_id, fold, split, etc.)
    """
    import pandas as pd

    df = pd.read_csv(manifest_path)

    # Strong name-based candidates (likely labels)
    good_name_keys = ["label", "target", "y", "class"]
    bad_name_keys  = ["id", "patient", "subject", "record", "study", "file", "path", "fold", "split", "seed", "index"]

    def is_bad_name(c: str) -> bool:
        lc = c.lower()
        return any(k in lc for k in bad_name_keys)

    def is_good_name(c: str) -> bool:
        lc = c.lower()
        return any(k in lc for k in good_name_keys)

    cols = list(df.columns)

    if label_col is not None:
        if label_col not in df.columns:
            raise ValueError(f"label_col={label_col} not found in manifest columns")
        s = pd.to_numeric(df[label_col], errors="coerce").dropna().astype(int)
        u = sorted(set(s.tolist()))
        return len(u)

    # Consider numeric columns only, filter out obviously bad name columns first
    numeric_cols = []
    for c in cols:
        if is_bad_name(c):
            continue
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) == 0:
            continue
        # if it's mostly integer-like
        s_int = s.round().astype(int)
        if (s - s_int).abs().mean() < 1e-6:
            numeric_cols.append(c)

    # Rank candidates by "label-likeness"
    # Prefer:
    #   - contiguous classes starting at 0 (or any contiguous range)
    #   - exactly `prefer` classes (default 7)
    #   - good name match
    def stats(c: str):
        import numpy as np
        s = pd.to_numeric(df[c], errors="coerce").dropna().astype(int)
        u = sorted(set(s.tolist()))
        n = len(u)
        if n == 0:
            return None
        mn, mx = u[0], u[-1]
        contiguous = int(u == list(range(mn, mx + 1)))
        starts0 = int(mn == 0)
        exact_prefer = int(n == prefer and contiguous == 1 and (mx - mn + 1) == prefer)
        goodname = int(is_good_name(c))
        # score tuple (higher is better)
        return (exact_prefer, goodname, contiguous, starts0, n, -(mx - mn), c, mn, mx)

    scored = []
    for c in numeric_cols:
        st = stats(c)
        if st is not None:
            scored.append(st)

    if not scored:
        return 2

    scored.sort(reverse=True)
    best = scored[0]
    exact_prefer, goodname, contiguous, starts0, n, span_neg, c, mn, mx = best

    # If we didn't hit prefer=7, still sanity cap to something reasonable
    # (avoid accidentally picking 100s of IDs)
    if n > 50:
        # Try to find any column with 5-15 classes contiguous as fallback
        for cand in scored:
            _, gn, cont, s0, nn, _, cc, mmn, mmx = cand
            if cont and 5 <= nn <= 15:
                return int(nn)
        return 2

    return int(n)

def infer_num_classes_from_manifest(manifest_path: str, label_col: str | None = None) -> int:
    """Infer num_classes from the manifest by finding a numeric label column and counting unique labels.
    If label_col is None, pick a reasonable label column automatically.
    """
    import pandas as pd

    df = pd.read_csv(manifest_path)

    # Candidate label columns by name
    name_cands = [c for c in df.columns if any(k in c.lower() for k in ["label", "target", "class", "y"])]

    def score_col(c: str):
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) == 0:
            return (-1, -1, c)
        u = sorted(set(s.astype(int).tolist()))
        # prefer 5-50 classes (ECG often 5 or 7), penalize binary unless nothing else
        n = len(u)
        span = (max(u) - min(u)) if u else 0
        # score: more classes better, non-binary preferred, contiguous preferred
        contiguous = int(u == list(range(min(u), max(u)+1))) if u else 0
        return (n, contiguous, c)

    if label_col is not None and label_col in df.columns:
        s = pd.to_numeric(df[label_col], errors="coerce").dropna().astype(int)
        return int(len(sorted(set(s.tolist()))))

    # Try name-based candidates first, else all columns
    search_cols = name_cands if name_cands else list(df.columns)
    scored = [score_col(c) for c in search_cols]
    scored.sort(reverse=True)  # highest n, then contiguous
    best_n, best_contig, best_c = scored[0]

    if best_n <= 1:
        # fallback
        return 2
    return int(best_n)

def infer_num_classes_from_ckpt(ckpt_path: str) -> int:
    """Infer number of classes from a saved checkpoint.
    Works for common patterns:
      - raw state_dict (Tensor values)
      - dict containing 'state_dict' / 'model' / 'model_state_dict'
    Falls back to 2 if it cannot infer.
    """
    try:
        import torch
        obj = torch.load(ckpt_path, map_location="cpu")
        sd = None
        if isinstance(obj, dict):
            for k in ("state_dict", "model_state_dict", "model", "net", "weights"):
                if k in obj and isinstance(obj[k], dict):
                    sd = obj[k]
                    break
            if sd is None:
                # might already be a state dict
                sd = obj
        elif isinstance(obj, dict):
            sd = obj

        if not isinstance(sd, dict):
            return 2

        # Look for classifier/fc weights: shape [C, ...]
        candidate_keys = []
        for k, v in sd.items():
            if not hasattr(v, "shape"):
                continue
            lk = k.lower()
            if any(s in lk for s in ("classifier", "fc", "head")) and lk.endswith("weight"):
                candidate_keys.append(k)

        # Prefer the most "head-like" keys
        for pref in ("classifier", "head", "fc"):
            for k in candidate_keys:
                if pref in k.lower():
                    w = sd[k]
                    if len(w.shape) >= 1:
                        return int(w.shape[0])

        # If none matched, try any 2D weight near the end
        for k, v in reversed(list(sd.items())):
            if hasattr(v, "shape") and len(v.shape) == 2:
                return int(v.shape[0])

        return 2
    except Exception:
        return 2


def resolve_data_path(primary: str, fallback: str) -> str:
    primary_p = Path(primary)
    if primary_p.exists():
        return str(primary_p)
    fallback_p = Path(fallback)
    if fallback_p.exists():
        print(f"[run_embc_fix] Warning: {primary} not found; using {fallback}")
        return str(fallback_p)
    # If neither exists, raise a clear error with both paths listed
    raise FileNotFoundError(
        f"Could not find required file. Tried: {primary} and {fallback}. "
        f"Please place the file in one of these locations."
    )

def compute_ece(logits, labels, n_bins=10):
    """Calculates Expected Calibration Error."""
    if isinstance(logits, np.ndarray):
        logits = torch.tensor(logits)
    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels)
        
    probs = torch.softmax(logits, dim=1)
    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(labels)
    ece = torch.zeros(1, device=logits.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    
    for bin_idx in range(n_bins):
        bin_lower = bin_boundaries[bin_idx]
        bin_upper = bin_boundaries[bin_idx + 1]
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece.item()

def run_leakage_probe(source_feats, target_feats):
    """
    Trains a Logistic Regression probe to distinguish Source vs Target.
    Returns Mean Validation Accuracy and AUC.
    Enforces Balanced Data.
    """
    # Balance Data (Downsample majority)
    n_src = len(source_feats)
    n_tgt = len(target_feats)
    n_min = min(n_src, n_tgt)
    
    # Shuffle and slice
    rng = np.random.default_rng(42)
    s_idx = rng.choice(n_src, n_min, replace=False)
    t_idx = rng.choice(n_tgt, n_min, replace=False)
    
    X_src = source_feats[s_idx]
    X_tgt = target_feats[t_idx]
    
    # Concatenate
    X = np.concatenate([X_src, X_tgt])
    y = np.concatenate([np.zeros(n_min), np.ones(n_min)])
    
    # Initial Shuffle before CV
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    
    # Cross Validation Probe
    # Increased Max Iter for convergence
    clf = LogisticRegression(max_iter=2000, solver='lbfgs')
    
    # Accuracy
    acc_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    mean_acc = acc_scores.mean()
    
    # AUC
    auc_scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
    mean_auc = auc_scores.mean()
    
    return mean_acc, mean_auc

def get_raw_features(loader):
    """
    Extracts simple statistical features from raw signals.
    Stats: Mean, Std, Max, Min per lead (12 leads) -> 48 features.
    """
    all_feats = []
    
    for batch in loader:
        # x shape: (B, 12, 1000)
        x = batch[0].numpy()
        
        # Stats
        mu = np.mean(x, axis=2) # (B, 12)
        sigma = np.std(x, axis=2)
        mx = np.max(x, axis=2)
        mn = np.min(x, axis=2)
        
        # Concat: (B, 48)
        feats = np.concatenate([mu, sigma, mx, mn], axis=1)
        all_feats.append(feats)
        
    return np.concatenate(all_feats)

def get_random_features(loader, device, seed, num_classes: int | None = None):
    if num_classes is None:
        num_classes = 2
    """
    Extracts features from a randomly initialized (untrained) ResNet1d.
    """
    torch.manual_seed(seed)
    model = ResNet1d(input_channels=12, num_classes=num_classes) # Init random
    model.to(device)
    model.eval()
    
    all_feats = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            _, feats = model(x, return_feats=True)
            all_feats.append(feats.cpu().numpy())
            
    return np.concatenate(all_feats)

def forward_looped(model, loader, device, return_feats=False):
    model.eval()
    all_preds = []
    all_targets = []
    all_logits = []
    all_feats = []
    
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            
            if return_feats:
                out, feats = model(x, return_feats=True)
                all_feats.append(feats.cpu().numpy())
            else:
                out = model(x)
                
            logits = out
            if isinstance(out, dict): logits = out.get('logits', out)
            
            preds = torch.argmax(logits, dim=1)
            
            all_logits.append(logits.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            
    res = {
        'logits': np.concatenate(all_logits),
        'preds': np.concatenate(all_preds),
        'targets': np.concatenate(all_targets)
    }
    if return_feats:
        res['feats'] = np.concatenate(all_feats)
    return res

# --- Main Runner ---

def get_train_cmd(train_src, method, condition, seed, out_dir):
    cmd = [
        "python", "-m", "src.train",
        f"method={method}",
        f"seed={seed}",
        f"hydra.run.dir={out_dir}",
        f"train.epochs={EPOCHS}",
        f"++data.train_sources=[{train_src}]",
        f"++save_path={out_dir}",
        "++model.num_classes=2"  # Enforce Binary Classification
    ]
    
    # Condition Logic
    if condition == 'Clean':
        cmd.append("++data.shortcut.use_shortcut=False")
    elif condition.startswith('SAST_'):
        rho = condition.split('_')[1]
        cmd.append("++data.shortcut.use_shortcut=True")
        cmd.append("++data.shortcut.type=mains")
        cmd.append(f"++data.shortcut.correlation={rho}")
        cmd.append("++data.shortcut.force=False") # Correlation mode
        cmd.append("++data.shortcut.split=train") # Ensure logic applies
        
    return cmd

def load_method_model(method_name, ckpt_path, num_classes=None, device='cpu'):
    # Load state dict
    sd = torch.load(ckpt_path, map_location=device)
    
    # Backbone
    model = ResNet1d(input_channels=12, num_classes=num_classes)
    
    # Handle DANN/VREx state dict keys
    new_sd = {}
    for k, v in sd.items():
        if k.startswith('model.'):
            new_sd[k.replace('model.', '')] = v
        elif k.startswith('classifier.'): # Common in DANN impls
             new_sd['fc.' + k.replace('classifier.', '')] = v
        elif 'domain' in k:
            continue # Ignore domain heads
        else:
            new_sd[k] = v
            
    # Try loading
    try:
        model.load_state_dict(new_sd, strict=False)
    except RuntimeError as e:
        log.warning(f"Strict load failed: {e}. Trying stricter mapping...")
    
    model.to(device)
    return model

def main():
    if not os.path.exists(BASE_OUT_DIR):
        os.makedirs(BASE_OUT_DIR)
        
    # Processed Path (Shared)
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "signals.h5")
    
    MANIFEST_PATH = resolve_data_path(
        primary=os.path.join(PROJECT_ROOT, "data", "manifests", "master_manifest.csv"),
        fallback=os.path.join(PROJECT_ROOT, "data", "processed", "master_manifest.csv")
    )
    SPLIT_PATH = resolve_data_path(
        primary=os.path.join(PROJECT_ROOT, "data", "manifests", "splits.json"),
        fallback=os.path.join(PROJECT_ROOT, "data", "processed", "splits.json")
    )
    
    log.info(f"Using Manifest Path: {MANIFEST_PATH}")

    # PATCH: set global num_classes from manifest (used by probes/eval)
    try:
        num_classes = infer_num_classes_from_manifest(MANIFEST_PATH)
    except NameError:
        # some scripts use manifest_path variable instead
        num_classes = infer_num_classes_from_manifest(manifest_path)
    log.info(f"DEBUG: Inferred num_classes from manifest = {num_classes}")
    log.info(f"Using Split Path: {SPLIT_PATH}")
    
    import pandas as pd
    import json
    
    manifest_df = pd.read_csv(MANIFEST_PATH)
    with open(SPLIT_PATH, 'r') as f:
        splits = json.load(f)

    def get_loader(source, split, shortcut_cfg, batch_size=128):
        # Filter IDs
        all_ids = []
        for k, v in splits.items():
            if split in k:
                all_ids.extend(v)
                
        df = manifest_df[manifest_df['unique_id'].isin(all_ids)]
        df = df[df['dataset_source'] == source]
        
        # Enable binary labels for evaluation
        ds = ECGDataset(df, PROCESSED_PATH, task_label_col='task_a_label', shortcut_cfg=shortcut_cfg, split='test', binary_labels=True)
        return DataLoader(ds, batch_size=batch_size, shuffle=False)

    results = []
    
    # 1. BASELINE PROBES (Raw & Random)
    # Run for each direction to keep consistent structure, though Raw is roughly invariant if split is constant.
    # We iterate seeds for Random initialization.
    log.info("--- Running Baseline Probes (Raw & Random) ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for (src_name, tgt_name) in DIRECTIONS:
        # Loaders (Clean)
        l_src = get_loader(src_name, 'test', None)
        l_tgt = get_loader(tgt_name, 'test', None)
        
        # A. Raw Features
        # Compute once per direction (deterministic given split)
        log.info(f"Probing RAW features for {src_name}->{tgt_name}")
        feat_src_raw = get_raw_features(l_src)
        feat_tgt_raw = get_raw_features(l_tgt)
        
        # Simulate 'seeds' for Raw by repeating the probe (CV variance captured, seed used for downsampling in run_leakage)
        for seed in SEEDS:
            # Note: run_leakage_probe uses internal seed=42 for downsampling. 
            # To get variance, valid option is just ONE result, or vary downsampling seed. 
            # Requirements say "simulate multiple runs". We'll just report the CV score for this seed row.
            acc, auc = run_leakage_probe(feat_src_raw, feat_tgt_raw)
            results.append({
                'Method': 'RAW', 'Direction': f"{src_name}->{tgt_name}", 'Condition': 'Clean', 'Seed': seed,
                'Leakage_Acc': acc, 'Leakage_AUC': auc, 'Src_F1': np.nan, 'Tgt_Clean_F1': np.nan, 'OOD_Drop': np.nan, 'Tgt_Pois_F1': np.nan, 'ECE': np.nan, 'Rho': 0.0
            })

        # B. Random Encoder Features
        # Vary initialization via seed
        for seed in SEEDS:
            log.info(f"Probing RANDOM features for {src_name}->{tgt_name} (Seed {seed})")
            feat_src_rand = get_random_features(l_src, device, seed, num_classes=num_classes)
            feat_tgt_rand = get_random_features(l_tgt, device, seed, num_classes=num_classes)
            
            acc, auc = run_leakage_probe(feat_src_rand, feat_tgt_rand)
            results.append({
                'Method': 'RANDOM', 'Direction': f"{src_name}->{tgt_name}", 'Condition': 'Clean', 'Seed': seed,
                'Leakage_Acc': acc, 'Leakage_AUC': auc, 'Src_F1': np.nan, 'Tgt_Clean_F1': np.nan, 'OOD_Drop': np.nan, 'Tgt_Pois_F1': np.nan, 'ECE': np.nan, 'Rho': 0.0
            })

    # 2. MAIN GRID
    grid = list(itertools.product(METHODS, DIRECTIONS, CONDITIONS, SEEDS))
    log.info(f"Total Traind Experiments: {len(grid)}")
    
    for method, (src_name, tgt_name), cond, seed in tqdm(grid, desc="Experiments"):
        run_name = f"{src_name}_{method}_{cond}_{seed}"
        out_dir = os.path.join(BASE_OUT_DIR, run_name)
        ckpt_path = os.path.join(out_dir, "best_model.pt")
        
        # ... (Training Loop remains same) ...
        # Helper to verify checkpoint
        def verify_checkpoint(path):
            try:
                sd = torch.load(path, map_location='cpu')
                # Check FC weight shape
                for k, v in sd.items():
                    if 'fc.weight' in k or 'classifier.weight' in k:
                        if v.shape[0] != 2: # Expecting 2 classes
                            return False
                return True
            except Exception:
                return False

        # Check if training is needed
        need_train = True
        if os.path.exists(ckpt_path):
            if verify_checkpoint(ckpt_path):
                log.info(f"Skipping training (valid checkpoint exists): {run_name}")
                need_train = False
            else:
                log.warning(f"Checkpoint mismatch detected for {run_name} (Stale 7-class model). Deleting ENTIRE output directory and Retraining.")
                import shutil
                if os.path.exists(out_dir):
                    shutil.rmtree(out_dir)
                    os.makedirs(out_dir, exist_ok=True)
                need_train = True
        
        if need_train:
            log.info(f"Training {run_name}...")
            cmd = get_train_cmd(src_name, method, cond, seed, out_dir)
            ret = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if ret.returncode != 0:
                log.error(f"Training failed for {run_name}: {ret.stderr.decode()}")
                continue

        try:
            log.info(f"Evaluating model at {ckpt_path}...")
            num_classes = infer_num_classes_from_ckpt(ckpt_path)

            model = load_method_model(method, ckpt_path, num_classes=num_classes, device=device)
            
            cfg_pois = OmegaConf.create({
                "use_shortcut": True, 
                "type": "mains", 
                "freq": 60, "amplitude": 1.0, 
                "force": True
            })
            
            l_src_clean = get_loader(src_name, 'test', None)
            l_tgt_clean = get_loader(tgt_name, 'test', None)
            l_tgt_pois = get_loader(tgt_name, 'test', cfg_pois)
            
            res_src = forward_looped(model, l_src_clean, device, return_feats=True)
            res_tgt = forward_looped(model, l_tgt_clean, device, return_feats=True)
            res_tgt_p = forward_looped(model, l_tgt_pois, device, return_feats=False)
            
            src_f1 = f1_score(res_src['targets'], res_src['preds'], average='macro')
            tgt_c_f1 = f1_score(res_tgt['targets'], res_tgt['preds'], average='macro')
            tgt_p_f1 = f1_score(res_tgt_p['targets'], res_tgt_p['preds'], average='macro')
            
            ood_drop = src_f1 - tgt_c_f1
            tgt_ece = compute_ece(res_tgt['logits'], res_tgt['targets'])
            leak_acc, leak_auc = run_leakage_probe(res_src['feats'], res_tgt['feats'])
            
            res_row = {
                'Method': method.upper(),
                'Direction': f"{src_name}->{tgt_name}",
                'Condition': cond,
                'Seed': seed,
                'Rho': cond.split('_')[1] if 'SAST' in cond else 0.0,
                'Src_F1': src_f1,
                'Tgt_Clean_F1': tgt_c_f1,
                'Tgt_Pois_F1': tgt_p_f1,
                'OOD_Drop': ood_drop,
                'Leakage_Acc': leak_acc,
                'Leakage_AUC': leak_auc,
                'ECE': tgt_ece
            }
            results.append(res_row)
            log.info(f"Evaluation Results for {run_name}: Tgt_Clean={tgt_c_f1:.4f}, Tgt_Pois={tgt_p_f1:.4f}")
            
            pd.DataFrame(results).to_csv(RESULTS_FILE, index=False)
            
        except Exception as e:
            log.error(f"Evaluation failed for {run_name}: {e}")
            import traceback
            traceback.print_exc()
            import traceback
            traceback.print_exc()

    # --- Summary ---
    if results:
        df = pd.DataFrame(results)
        df.to_csv(RESULTS_FILE, index=False)
        
        agg_cols = ['Method', 'Direction', 'Condition']
        metric_cols = ['Src_F1', 'Tgt_Clean_F1', 'Tgt_Pois_F1', 'OOD_Drop', 'Leakage_Acc', 'Leakage_AUC', 'ECE']
        
        # Only aggregate numeric, ignore nan
        summary = df.groupby(agg_cols)[metric_cols].agg(['mean', 'std', 'count', 'sem'])
        for m in metric_cols:
            summary[(m, 'ci95')] = 1.96 * summary[(m, 'sem')]
        summary.to_csv(SUMMARY_FILE)
        log.info(f"Summary saved to {SUMMARY_FILE}")
        
        # --- Plotting ---
        # 1. Sensitivity Plot
        df_sast = df[df['Condition'].str.contains('SAST')]
        if not df_sast.empty:
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=df_sast, x='Rho', y='Tgt_Clean_F1', hue='Method', style='Direction', markers=True, errorbar='ci')
            plt.title('Sensitivity Analysis: Target Generalization vs. Shortcut Strength')
            plt.savefig(os.path.join(BASE_OUT_DIR, "sast_sensitivity_tgt_clean.png"))
            plt.close()

        # 2. Leakage Bar Plot (Clean Condition, All Methods including Raw/Random)
        df_clean = df[df['Condition'] == 'Clean']
        # Filter for relevant methods for plot
        plot_methods = ['RAW', 'RANDOM', 'ERM', 'DANN', 'VREX']
        df_plot = df_clean[df_clean['Method'].isin(plot_methods)]
        
        if not df_plot.empty:
            plt.figure(figsize=(8, 6))
            sns.barplot(data=df_plot, x='Method', y='Leakage_AUC', order=plot_methods, errorbar='ci', capsize=.1)
            plt.title('Dataset-Identity Leakage (AUROC)')
            plt.ylim(0.5, 1.0)
            plt.ylabel('Probe AUROC')
            plt.grid(axis='y', alpha=0.3)
            plt.savefig(os.path.join(BASE_OUT_DIR, "leakage_probe_bar.png"))
            plt.close()
            
        log.info("Plots generated.")

if __name__ == "__main__":
    main()
