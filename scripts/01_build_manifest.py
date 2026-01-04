"""
Script: 01_build_manifest.py
Description: Scans raw datasets, harmonizes labels, generates Train/Val/Test splits, and saves to master_manifest.csv.
Refined path logic for PTB-XL and Header parsing for Chapman.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import wfdb
import ast
import json
import logging
import argparse
import re
from omegaconf import OmegaConf
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold

# Ensure src is in path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))
try:
    from src.data.mappings import PTBXL_MAPPING, CHAPMAN_MAPPING, TASK_A_LABELS
except ImportError:
    sys.path.append(os.path.abspath("."))
    from src.data.mappings import PTBXL_MAPPING, CHAPMAN_MAPPING, TASK_A_LABELS

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

def parse_header_comments(hea_path):
    """Parse .hea file for PhysioNet style comments (# key: value)."""
    meta = {}
    try:
        with open(hea_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    # Examples: 
                    # # Age: 50
                    # # Sex: Male
                    # # Dx: 426783006, 164889003
                    parts = line.strip().lstrip('#').split(':')
                    if len(parts) >= 2:
                        key = parts[0].strip()
                        val = parts[1].strip()
                        meta[key] = val
    except Exception as e:
        log.warning(f"Error reading header {hea_path}: {e}")
    return meta

def process_ptbxl(config):
    """Process PTB-XL database using provided filename columns."""
    log.info("Processing PTB-XL...")
    root = config.paths.ptbxl
    db_path = os.path.join(root, 'ptbxl_database.csv')
    
    if not os.path.exists(db_path):
         log.error(f"PTB-XL database not found at {db_path}")
         return pd.DataFrame()
         
    df = pd.read_csv(db_path, index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load SCP Statements for mapping superclasses if available
    scp_path = os.path.join(root, 'scp_statements.csv')
    code_to_class = {}
    if os.path.exists(scp_path):
        scp_df = pd.read_csv(scp_path, index_col=0)
        code_to_class = scp_df['diagnostic_class'].to_dict()

    clean_records = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="PTB-XL"):
        # USE PROVIDED FILENAME COLUMN
        # ptbxl_database.csv has 'filename_hr' e.g. "records500/00000/00001_hr"
        rel_path = row['filename_hr']
        full_path = os.path.join(root, rel_path + ".dat") # .dat might not be in the string, usually string is base path
        
        # Check if extension is needed.
        # Usually WFDB python tools expect the base name without extension for .dat/.hea pair
        # But for checking file existence, we check .dat
        if not os.path.exists(full_path) and not os.path.exists(os.path.join(root, rel_path)):
             # Fallback to LR if HR missing
             if 'filename_lr' in row:
                 rel_path = row['filename_lr']
                 full_path = os.path.join(root, rel_path + ".dat")
                 if not os.path.exists(full_path):
                     continue
             else:
                 continue

        # Label Logic
        current_labels = []
        for code in row.scp_codes.keys():
            if code in code_to_class:
                sclass = code_to_class[code]
                if sclass in PTBXL_MAPPING:
                    current_labels.append(PTBXL_MAPPING[sclass])
            elif code in PTBXL_MAPPING:
                current_labels.append(PTBXL_MAPPING[code])
        
        # Priority Logic
        if 1 in current_labels: task_a = 1 # AFIB
        elif 2 in current_labels: task_a = 2 # GSVT
        elif 5 in current_labels: task_a = 5 # PVC
        elif 4 in current_labels: task_a = 4 # ST
        elif 3 in current_labels: task_a = 3 # SB
        elif 0 in current_labels: task_a = 0 # NSR
        else: task_a = 6 # OTHER
            
        clean_records.append({
            'unique_id': f"ptbxl_{idx}",
            'dataset_source': 'ptbxl',
            'original_path': rel_path,
            'sampling_rate': 500 if 'records500' in rel_path else 100,
            'patient_id': f"ptbxl_pat_{int(row['patient_id'])}",
            'age': row['age'] if not pd.isna(row['age']) else 0,
            'sex': 0 if row['sex'] == 0 else 1,
            'task_a_label': task_a,
            'raw_labels': str(list(row.scp_codes.keys()))
        })
        
    return pd.DataFrame(clean_records)

def process_chapman(config):
    """Process Chapman using .hea file parsing strategy."""
    log.info("Processing Chapman...")
    root = config.paths.chapman
    records_dir = os.path.join(root, 'WFDBRecords')
    
    clean_records = []
    
    # Recursively find all .hea files
    hea_files = glob.glob(os.path.join(records_dir, "**", "*.hea"), recursive=True)
    
    for hea_path in tqdm(hea_files, desc="Chapman Headers"):
        # Path logic
        rel_path = os.path.relpath(hea_path, root).replace('.hea', '') # Remove ext because WFDB tools add it
        basename = os.path.basename(rel_path)
        
        # Parse Header
        meta = parse_header_comments(hea_path)
        
        # Extract Dx (Diagnosis)
        # Format usually "# Dx: 1234, 5678"
        dx_str = meta.get('Dx', '')
        dx_codes = [c.strip() for c in dx_str.split(',') if c.strip()]
        
        # Extract Demographics
        age = meta.get('Age', 0)
        try: age = float(age)
        except: age = 0
        
        sex_str = meta.get('Sex', 'Male')
        sex = 0 if sex_str.startswith('M') or sex_str.startswith('m') else 1
        
        # Label Logic
        task_a = 6 # Default
        
        # Priority: AFIB > GSVT > PVC > ST > SB > NSR
        mapped_codes = []
        for code in dx_codes:
            if code in CHAPMAN_MAPPING:
                mapped_codes.append(CHAPMAN_MAPPING[code])
        
        if 1 in mapped_codes: task_a = 1
        elif 2 in mapped_codes: task_a = 2
        elif 5 in mapped_codes: task_a = 5
        elif 4 in mapped_codes: task_a = 4
        elif 3 in mapped_codes: task_a = 3
        elif 0 in mapped_codes: task_a = 0
        
        clean_records.append({
            'unique_id': f"chapman_{basename}",
            'dataset_source': 'chapman',
            'original_path': rel_path,
            'sampling_rate': 500,
            'patient_id': f"chapman_pat_{basename}", # Usually 1 record per patient in this set
            'age': age,
            'sex': sex,
            'task_a_label': task_a,
            'raw_labels': dx_str
        })
        
    return pd.DataFrame(clean_records)

def process_mitbih(config):
    """Process MIT-BIH."""
    log.info("Processing MIT-BIH...")
    root = config.paths.mitbih
    records = []
    
    dat_files = glob.glob(os.path.join(root, "*.dat"))
    for f in dat_files:
        bname = os.path.basename(f).replace('.dat', '')
        # Must have .atr
        if not os.path.exists(os.path.join(root, bname + '.atr')):
            continue
            
        records.append({
            'unique_id': f"mitbih_{bname}",
            'dataset_source': 'mitbih',
            'original_path': bname, # Just basename for MIT-BIH usually
            'sampling_rate': 360,
            'patient_id': f"mitbih_{bname}", 
            'age': 0, # Not easily available in filename, extract from header if needed, but not critical
            'sex': 0,
            'task_a_label': 6, # Windowing defines this
            'raw_labels': "ATR"
        })
        
    return pd.DataFrame(records)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/main.yaml')
    args = parser.parse_args()

    main_cfg = OmegaConf.load(args.config)
    data_cfg_path = os.path.join('config', 'data', 'default.yaml')
    if os.path.exists(data_cfg_path):
        data_cfg = OmegaConf.load(data_cfg_path)
        main_cfg = OmegaConf.merge(main_cfg, data_cfg)
    
    # Run
    df_ptb = process_ptbxl(main_cfg)
    df_chap = process_chapman(main_cfg)
    df_mit = process_mitbih(main_cfg)
    
    master = pd.concat([df_ptb, df_chap, df_mit], ignore_index=True)
    
    log.info(f"Total Records: {len(master)}")
    log.info(f"By Source:\n{master['dataset_source'].value_counts()}")
    
    # Validate Labels
    log.info("Class Counts (Task A):")
    log.info(master['task_a_label'].value_counts().sort_index())
    
    # Split Generation
    log.info("Generating Splits...")
    master['split'] = 'train'
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=main_cfg.seed)
    
    for ds in master['dataset_source'].unique():
        sub_df = master[master['dataset_source'] == ds]
        y = sub_df['task_a_label'].values
        groups = sub_df['patient_id'].values
        idxs = sub_df.index.values
        
        folds = list(splitter.split(idxs, y, groups))
        
        # Test: Folds 0,1 (20%)
        # Val: Fold 2 (10%)
        # Train: Rest
        
        test_idxs = np.concatenate([folds[0][1], folds[1][1]])
        val_idxs = folds[2][1]
        
        master.loc[idxs[test_idxs], 'split'] = 'test'
        master.loc[idxs[val_idxs], 'split'] = 'val'
    
    out_path = main_cfg.paths.manifest_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    master.to_csv(out_path, index=False)
    log.info(f"Saved manifest to {out_path}")
    
    splits = {
        'train': master[master['split']=='train']['unique_id'].tolist(),
        'val': master[master['split']=='val']['unique_id'].tolist(),
        'test': master[master['split']=='test']['unique_id'].tolist()
    }
    with open(os.path.join(os.path.dirname(out_path), 'splits.json'), 'w') as f:
        json.dump(splits, f)

if __name__ == "__main__":
    main()
