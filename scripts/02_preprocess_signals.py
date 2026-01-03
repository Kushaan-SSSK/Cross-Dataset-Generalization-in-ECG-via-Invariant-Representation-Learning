
import argparse
import os
import logging
import numpy as np
import pandas as pd
import wfdb
import h5py
from tqdm import tqdm
from scipy import signal
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_signal(record_path, dataset_source):
    """
    Load ECG signal using wfdb or scipy.io depending on format.
    Returns:
        signal (np.array): (n_samples, n_channels)
        fs (int): Sampling frequency
    """
    try:
        # MIT-BIH and PTB-XL are compatible with wfdb.rdsamp
        # Chapman (if .mat converted or just headers) might vary, but valid WFDB files allow rdsamp
        
        # Check if file exists - the record_path comes from original_path which is relative
        # We need to construct full path based on dataset structure if needed, 
        # but manifest builder should have ensured relative paths are correct for WFDB reading
        # WFDB reader needs path WITHOUT extension usually if header is separate, or full if binary?
        # Actually usually it takes base name.
        
        # However, for Chapman we used .hea parsing.
        # Let's try wfdb.rdsamp first.
        
        # NOTE: rdsamp argument 'record_name' should not include extension for most wfdb formats
        # But if we have direct paths like "path/to/123", it works.
        
        # Clean path: remove extension if present (like .dat or .mat or .hea)
        base_path = os.path.splitext(record_path)[0]
        
        data, fields = wfdb.rdsamp(base_path)
        return data, fields['fs']
        
    except Exception as e:
        # Fallback for specific Chapman structure if standard WFDB fails or if it's a mat file not linked to hea correctly
        # But manifest builder checked for .hea presence.
        raise ValueError(f"Failed to load record {record_path}: {e}")

def apply_filter(sig, fs, lowcut=0.5, highcut=50.0, order=3):
    """
    Apply Bandpass Filter using Butterworth design.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = min(highcut, nyquist - 1.0) / nyquist # Ensure strictly < 1
    
    if high <= low:
        # Fallback if range is invalid
        high = 0.99
    
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_sig = signal.filtfilt(b, a, sig, axis=0) # Filter along time axis
    return filtered_sig

def resample_signal(sig, original_fs, target_fs):
    """
    Resample signal to target frequency.
    """
    if original_fs == target_fs:
        return sig
    
    num_samples = int(len(sig) * target_fs / original_fs)
    resampled_sig = signal.resample(sig, num_samples, axis=0)
    return resampled_sig

def normalize_signal(sig):
    """
    Z-score normalization (zero mean, unit variance) per lead.
    """
    # Epsilon to avoid division by zero
    eps = 1e-8
    mean = np.mean(sig, axis=0)
    std = np.std(sig, axis=0)
    return (sig - mean) / (std + eps)

def process_record(row, root_path, target_fs=100):
    """
    Process a single record: Load -> Resample -> Filter -> Normalize.
    """
    unique_id = row['unique_id']
    dataset_source = row['dataset_source']
    rel_path = row['original_path']
    
    # Construct full path
    # For PTB-XL, original_path is like "records500/..." relative to PTB root
    # For Chapman, it's relative to Chapman root
    # For MIT-BIH, it's relative to MIT-BIH root
    
    # We need to resolve the correct root based on source
    # This requires the config to pass these roots OR we assume a strict structure.
    # Ideally we pass a dictionary of roots.
    
    # Actually, the manifest `original_path` was constructed relative to the *dataset specific root*.
    # So we need to join it with the correct dataset root.
    
    full_path = ""
    # We will rely on the paths passed via config/args
    
    return None # Placeholder for structure

def crop_or_pad(sig, target_len=1000):
    """
    Fixed center crop or zero pad to target length.
    """
    length = sig.shape[0]
    if length == target_len:
        return sig
    
    if length > target_len:
        start = (length - target_len) // 2
        return sig[start:start+target_len, :]
    else:
        # Pad with zeros at the end
        pad_len = target_len - length
        # Pad width: ((before, after), (channel_before, channel_after))
        # Only pad time axis
        return np.pad(sig, ((0, pad_len), (0, 0)), 'constant')

def main():
    parser = argparse.ArgumentParser(description="Preprocess ECG signals")
    parser.add_argument("--config", type=str, default="config/data/default.yaml", help="Path to config file")
    args = parser.parse_args()

    conf = OmegaConf.load(args.config)
    
    # Paths
    # root_dir = conf.paths.root  # This key doesnt exist in data config
    manifest_path = conf.paths.manifest_path
    processed_path = conf.paths.processed_path
    target_fs = conf.sampling_rate
    
    # Dataset specific roots from config
    ptbxl_root = conf.paths.ptbxl
    chapman_root = conf.paths.chapman
    mitbih_root = conf.paths.mitbih
    
    source_roots = {
        'ptbxl': ptbxl_root,
        'chapman': chapman_root,
        'mitbih': mitbih_root
    }

    # Load Manifest
    logger.info(f"Loading manifest from {manifest_path}")
    df = pd.read_csv(manifest_path)
    
    # Ensure output dir exists
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    
    # Open HDF5 file
    logger.info(f"Writing to {processed_path}")
    count_success = 0
    count_fail = 0
    
    target_length = 10 * target_fs # 10s * 100Hz = 1000 samples
    
    with h5py.File(processed_path, 'w') as h5f:
        # Loop through manifest
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Signals"):
            unique_id = row['unique_id']
            source = row['dataset_source']
            rel_path = row['original_path']
            
            # Resolve Full Path
            if source not in source_roots:
                logger.warning(f"Unknown source {source} for {unique_id}. Skipping.")
                count_fail += 1
                continue
                
            record_root = source_roots[source]
            full_path = os.path.normpath(os.path.join(record_root, rel_path))
            
            try:
                # 1. Load
                sig, fs = load_signal(full_path, source)
                
                # Handle NaNs
                if np.isnan(sig).any():
                    df_sig = pd.DataFrame(sig)
                    df_sig = df_sig.ffill().bfill()
                    sig = df_sig.values
                    sig = np.nan_to_num(sig)

                # 2. Resample
                sig_resampled = resample_signal(sig, fs, target_fs)
                
                # 3. Crop/Pad (Standardize Length for PTB/Chapman)
                if source in ['ptbxl', 'chapman']:
                    sig_resampled = crop_or_pad(sig_resampled, target_length)
                
                # 4. Filter
                sig_filtered = apply_filter(sig_resampled, target_fs)
                
                # 5. Normalize
                sig_norm = normalize_signal(sig_filtered)
                
                # 6. Save to HDF5
                h5f.create_dataset(unique_id, data=sig_norm, compression="gzip", compression_opts=4)
                
                count_success += 1
                
            except Exception as e:
                logger.error(f"Error processing {unique_id}: {e}")
                count_fail += 1
                if count_fail >= 10:
                    logger.error("Too many failures. Stopping.")
                    break
    
    logger.info(f"Processing Complete. Success: {count_success}, Failures: {count_fail}")

if __name__ == "__main__":
    main()
