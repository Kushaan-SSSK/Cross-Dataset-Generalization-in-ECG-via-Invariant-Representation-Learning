
import os
import sys
import torch
import numpy as np
import pandas as pd
import logging
import hydra
from omegaconf import OmegaConf
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, roc_auc_score

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import ECGDataset
from torch.utils.data import DataLoader
from src.models.resnet1d import ResNet1d

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Configuration ---
# Use the same configuration base as run_embc_fix
SEEDS = [0, 1, 2, 3, 4]
METHODS = ['erm', 'dann', 'vrex']
DIRECTIONS = [('ptbxl', 'chapman'), ('chapman', 'ptbxl')]
# For Swap Test, we need:
# Clean Checkpoint: Condition="Clean"
# Poisoned Checkpoint: Condition="SAST_0.9" (Standard Poisoning Level for contrast)
POISON_COND = "SAST_0.9" 

BASE_OUT_DIR = "outputs/embc_fix"
SWAP_RESULTS_FILE = os.path.join(BASE_OUT_DIR, "swap_test_results.csv")
SWAP_SUMMARY_FILE = os.path.join(BASE_OUT_DIR, "swap_test_summary.csv")

# --- Helper Functions ---

def load_components(ckpt_path, num_classes=5, device='cpu'):
    """
    Loads a checkpoint and separates Encoder and Classification Head weights.
    Returns: (encoder_state_dict, head_state_dict)
    """
    if not os.path.exists(ckpt_path):
        return None, None
        
    full_state = torch.load(ckpt_path, map_location=device)
    
    encoder_sd = {}
    head_sd = {}
    
    # Heuristic mapping based on ResNet1d structure
    # Encoder: everything except 'fc'
    # Head: 'fc'
    
    # Note: DANN/VREx might have 'classifier.' prefix or 'model.' prefix
    # We strip 'model.' first
    
    clean_state = {}
    for k, v in full_state.items():
        if k.startswith('model.'):
            clean_state[k.replace('model.', '')] = v
        else:
            clean_state[k] = v
            
    for k, v in clean_state.items():
        if 'fc' in k or 'classifier' in k:
            # Map to standard 'fc' name if possible, or keep as is for specific architectures
            # Our ResNet1d uses 'fc'
            if k.startswith('classifier.'):
                head_sd['fc.' + k.replace('classifier.', '')] = v
            else:
                head_sd[k] = v
        elif 'domain' in k:
            continue # Ignore domain heads
        else:
            encoder_sd[k] = v
            
    if not encoder_sd or not head_sd:
        log.warning(f"Failed to split weights for {ckpt_path}. Keys found: {list(clean_state.keys())[:5]}")
        return None, None
        
    return encoder_sd, head_sd

def build_swapped_model(encoder_sd, head_sd, num_classes=7, device='cpu'):
    """
    Constructs a ResNet1d model and loads distinct encoder and head weights.
    """
    model = ResNet1d(input_channels=12, num_classes=num_classes)
    
    # Load Encoder
    model.load_state_dict(encoder_sd, strict=False)
    
    # Load Head
    # Strict=False is risky for Head if we miss keys, but necessary if encoder keys are missing.
    # We can do it in two passes or just load carefully.
    model.load_state_dict(head_sd, strict=False)
    
    model.to(device)
    model.eval()
    return model

def forward_eval(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            
            logits = model(x)
            if isinstance(logits, dict): logits = logits.get('logits', logits)
            
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            
    return np.concatenate(all_preds), np.concatenate(all_targets)

# --- Main Runner ---

def main():
    if not os.path.exists(BASE_OUT_DIR):
        log.error(f"Base output directory {BASE_OUT_DIR} does not exist. Run main experiments first.")
        # Create it just in case, but probably means no checkpoints
        os.makedirs(BASE_OUT_DIR, exist_ok=True)

    # Setup Paths
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "signals.h5")
    MANIFEST_PATH = os.path.join(PROJECT_ROOT, "data", "manifests", "master_manifest.csv")
    SPLIT_PATH = os.path.join(PROJECT_ROOT, "data", "manifests", "splits.json")
    
    import json
    manifest_df = pd.read_csv(MANIFEST_PATH)
    with open(SPLIT_PATH, 'r') as f:
        splits = json.load(f)

    def get_loader(source, split):
        # Clean Test Loader
        all_ids = []
        for k, v in splits.items():
            if split in k:
                all_ids.extend(v)
        df = manifest_df[manifest_df['unique_id'].isin(all_ids)]
        df = df[df['dataset_source'] == source]
        ds = ECGDataset(df, PROCESSED_PATH, task_label_col='task_a_label', shortcut_cfg=None, split='test')
        return DataLoader(ds, batch_size=128, shuffle=False)

    results = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Only iterate direction and method and seed
    # Condition is fixed: combining Clean vs SAST_0.9
    
    for (src_name, tgt_name) in DIRECTIONS:
        log.info(f"Processing Direction: {src_name} -> {tgt_name}")
        
        # We need the Target Test Loader (Clean) for evaluation
        l_tgt_clean = get_loader(tgt_name, 'test')
        
        for method in METHODS:
            for seed in SEEDS:
                # 1. Locate Checkpoints
                dir_clean = os.path.join(BASE_OUT_DIR, f"{src_name}_{method}_Clean_{seed}")
                dir_pois = os.path.join(BASE_OUT_DIR, f"{src_name}_{method}_{POISON_COND}_{seed}")
                
                ckpt_clean = os.path.join(dir_clean, "best_model.pt")
                ckpt_pois = os.path.join(dir_pois, "best_model.pt")
                
                if not (os.path.exists(ckpt_clean) and os.path.exists(ckpt_pois)):
                    log.warning(f"Missing checkpoints for {method} seed {seed} ({src_name}->{tgt_name}). Skipping.")
                    continue
                    
                # 2. Load and Split Weights
                enc_c, head_c = load_components(ckpt_clean, device=device)
                enc_p, head_p = load_components(ckpt_pois, device=device)
                
                if not (enc_c and head_c and enc_p and head_p):
                    continue
                    
                # 3. Swap Combinations
                combinations = [
                    ('Clean', 'Clean', enc_c, head_c),
                    ('Clean', 'Poison', enc_c, head_p),
                    ('Poison', 'Clean', enc_p, head_c),
                    ('Poison', 'Poison', enc_p, head_p)
                ]
                
                for enc_name, head_name, enc_w, head_w in combinations:
                    # Build Model
                    model = build_swapped_model(enc_w, head_w, device=device)
                    
                    # Evaluate
                    preds, targets = forward_eval(model, l_tgt_clean, device)
                    
                    # Metrics
                    f1 = f1_score(targets, preds, average='macro')
                    
                    res_row = {
                        'Method': method.upper(),
                        'Direction': f"{src_name}->{tgt_name}",
                        'Seed': seed,
                        'Encoder_Source': enc_name,
                        'Head_Source': head_name,
                        'Target_Macro_F1': f1
                    }
                    results.append(res_row)
                    
                # Intermediate Save
                pd.DataFrame(results).to_csv(SWAP_RESULTS_FILE, index=False)

    # --- Summary & Plotting ---
    if results:
        df = pd.DataFrame(results)
        
        # Aggregate
        # Group by Method and Swap Config (Direction aggregated or separate?)
        # Paper usually asks for "Method" summary, implying aggregation over directions if not specified.
        # But directions might behave differently. Let's aggregate over Direction/Seed to get Method/Config stats.
        
        agg_cols = ['Method', 'Encoder_Source', 'Head_Source']
        summary = df.groupby(agg_cols)['Target_Macro_F1'].agg(['mean', 'std', 'count', 'sem'])
        summary.reset_index(inplace=True)
        summary['ci95'] = 1.96 * summary['sem']
        
        summary.to_csv(SWAP_SUMMARY_FILE)
        log.info(f"Summary saved to {SWAP_SUMMARY_FILE}")
        
        # Plot
        # Create a "Config" column for x-axis
        # Label: E_{enc}+H_{head}
        summary['Config'] = "E_" + summary['Encoder_Source'] + " + H_" + summary['Head_Source']
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=summary, x='Config', y='mean', hue='Method', yerr=summary['ci95'], capsize=.1)
        plt.title('Encoder-Head Decoupling (Swap Test): Target Generalization')
        plt.ylabel('Target Macro-F1 (Clean)')
        plt.xlabel('Model Configuration')
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(os.path.join(BASE_OUT_DIR, "swap_test_plot.png"))
        log.info(f"Plot saved to swap_test_plot.png")
        
        # --- Interpretation ---
        log.info("\n--- AUTOMATED INTERPRETATION ---")
        for method in summary['Method'].unique():
            sub = summary[summary['Method'] == method].set_index('Config')
            try:
                base = sub.loc['E_Clean + H_Clean', 'mean']
                e_poison = sub.loc['E_Poison + H_Clean', 'mean']
                h_poison = sub.loc['E_Clean + H_Poison', 'mean']
                both = sub.loc['E_Poison + H_Poison', 'mean']
                
                delta_enc = e_poison - base
                delta_head = h_poison - base
                
                log.info(f"Method {method}:")
                log.info(f"  Base F1: {base:.3f}")
                log.info(f"  Encoder Effect (E_P+H_C): {delta_enc:+.3f}")
                log.info(f"  Head Effect    (E_C+H_P): {delta_head:+.3f}")
                
                if abs(delta_enc) > 2 * abs(delta_head):
                    log.info("  -> DOMINANT FACTOR: ENCODER (Representation Poisoning)")
                elif abs(delta_head) > 2 * abs(delta_enc):
                    log.info("  -> DOMINANT FACTOR: HEAD (Shortcut Exploitation in Classifier)")
                else:
                    log.info("  -> BOTH contribute (Coupled Poisoning)")
            except KeyError:
                log.warning(f"  Could not compute full interpretation for {method} (missing configs)")

if __name__ == "__main__":
    main()
