
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset
import numpy as np

class ECGDataset(Dataset):
    """
    ECG Dataset reading from HDF5 processed file.
    """
    def __init__(self, manifest_df, hdf5_path, task_label_col='task_a_label', shortcut_cfg=None, split='train'):
        """
        Args:
            manifest_df (pd.DataFrame): DataFrame containing 'unique_id' and labels.
            hdf5_path (str): Path to the processed .h5 file.
            task_label_col (str): Column name for the target label.
            shortcut_cfg (DictConfig): Configuration for shortcut injection.
            split (str): 'train', 'val', or 'test'.
        """
        # Validate keys against HDF5
        with h5py.File(hdf5_path, 'r') as f:
            existing_keys = set(f.keys())
        
        initial_len = len(manifest_df)
        self.manifest_df = manifest_df[manifest_df['unique_id'].isin(existing_keys)].reset_index(drop=True)
        final_len = len(self.manifest_df)
        
        if initial_len != final_len:
            print(f"Warning: Dropped {initial_len - final_len} records missing from HDF5.")
            
        self.hdf5_path = hdf5_path
        self.task_label_col = task_label_col
        self.shortcut_cfg = shortcut_cfg
        self.split = split
        self.h5_file = None

    def __len__(self):
        return len(self.manifest_df)
        
    def _add_shortcut(self, signal):
        """Add 60Hz sinusoidal noise."""
        if self.shortcut_cfg is None: return signal
        
        freq = self.shortcut_cfg.freq
        amp = self.shortcut_cfg.amplitude
        fs = 100 # Assumed from preprocessing config
        
        t = np.arange(signal.shape[0]) / fs
        noise = amp * np.sin(2 * np.pi * freq * t)
        
        # Add to Lead I (Index 0)
        signal[:, 0] += noise
        return signal

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.hdf5_path, 'r')

        row = self.manifest_df.iloc[idx]
        unique_id = row['unique_id']
        label = row[self.task_label_col]
        
        # Domain Mapping
        source = row['dataset_source']
        # 0: PTB-XL, 1: Chapman
        domain_map = {'ptbxl': 0, 'chapman': 1}
        domain = domain_map.get(source, 0) # Default to 0 if unknown

        # Retrieve signal
        try:
            signal = self.h5_file[unique_id][()]  # Read into numpy array
        except KeyError:
            raise KeyError(f"ID {unique_id} not found in {self.hdf5_path}")
            
        # --- Synthetic Shortcut Logic ---
        if self.shortcut_cfg and self.shortcut_cfg.use_shortcut:
            # Protocol: 
            # TRAIN: Inject correlation (e.g. 90% in Abnormal, 10% in Normal)
            # TEST/VAL: Clean (0% injection) to measure reliance drop
            
            inject = False
            if self.split == 'train':
                p_corr = self.shortcut_cfg.correlation
                
                # Check target label (Assuming Binary: 0=Normal, >0=Abnormal for simplicity of this benchmark)
                # Or specific class. Let's assume Correlation targets Abnormal (non-zero)
                is_abnormal = (label != 0) 
                
                rng = np.random.default_rng(seed=idx) # Deterministic per sample for reproducibility
                
                if is_abnormal:
                    # Inject with high probability (p_corr)
                    if rng.random() < p_corr:
                        inject = True
                else: 
                    # Normal: Inject with low probability (1 - p_corr)
                    if rng.random() < (1 - p_corr):
                        inject = True
            
            if inject:
                signal = self._add_shortcut(signal)

        # Signal shape: (L, C) -> Transpose to (C, L) for PyTorch Conv1d
        if signal.shape[0] > signal.shape[1]: 
             signal = signal.transpose(1, 0)
        
        # Convert to tensor
        signal_tensor = torch.tensor(signal, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        domain_tensor = torch.tensor(domain, dtype=torch.long)

        # Return (x, y, d)
        return signal_tensor, label_tensor, domain_tensor

    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()
