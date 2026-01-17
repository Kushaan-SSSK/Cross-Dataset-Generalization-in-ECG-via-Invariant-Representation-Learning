
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset
import numpy as np

from src.const import LABEL_COL, NUM_CLASSES, CLASS_NAMES
from src.sast import SASTProtocol
from omegaconf import OmegaConf

class ECGDataset(Dataset):
    """
    ECG Dataset reading from HDF5 processed file.
    Enforces strict 7-class task definition.
    """
    def __init__(self, manifest_df, hdf5_path, task_label_col=LABEL_COL, shortcut_cfg=None, split='train'):
        """
        Args:
            manifest_df (pd.DataFrame): DataFrame containing 'unique_id' and labels.
            hdf5_path (str): Path to the processed .h5 file.
            task_label_col (str): Column name for the target label. Default: src.const.LABEL_COL
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
            # print(f"Warning: Dropped {initial_len - final_len} records missing from HDF5.")
            pass
            
        self.hdf5_path = hdf5_path
        self.task_label_col = task_label_col
        self.shortcut_cfg = shortcut_cfg
        self.split = split
        self.h5_file = None
        
        # --- Runtime Label Validation ---
        unique_labels = self.manifest_df[self.task_label_col].unique()
        if not set(unique_labels).issubset(set(CLASS_NAMES)):
             raise ValueError(f"CRITICAL: Found labels outside expected set {CLASS_NAMES}. Found: {unique_labels}. Fix manifest mapping.")
        
        # --- SAST Protocol ---
        if shortcut_cfg:
             if 'split' not in shortcut_cfg: 
                 OmegaConf.set_struct(shortcut_cfg, False)
                 shortcut_cfg.split = split
        
        self.sast = SASTProtocol(shortcut_cfg if shortcut_cfg else {})
        
    def __len__(self):
        return len(self.manifest_df)
        


    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.hdf5_path, 'r')

        row = self.manifest_df.iloc[idx]
        unique_id = row['unique_id']
        label = row[self.task_label_col]
        
        # Binary Mapping Logic REMOVED. Enforcing 7-class.
        
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
            
        # --- SAST Injection ---
        signal = self.sast.inject_single(signal, label=label, unique_id=unique_id)

        # shape (C, L)
        if signal.shape[0] > signal.shape[1]: 
             signal = signal.transpose(1, 0)
        
        # --- Notch Filter (Frequency Augmentation) ---
        if getattr(self.shortcut_cfg, 'apply_notch', False):
            # Apply 60Hz (source) and 40Hz (alias) notch? 
            # Paper says 40Hz alias is the problem.
            # Let's filter 40Hz.
            from scipy.signal import iirnotch, dlsim
            fs = 100.0
            f0 = 40.0 # Frequency to remove
            Q = 30.0  # Quality factor
            b, a = iirnotch(f0, Q, fs)
            
            # Apply to all channels
            # signal is (C, L) numpy array here? No, earlier valid was numpy.
            # _add_shortcut returns numpy.
            filtered = []
            for ch in range(signal.shape[0]):
                # signal[ch] is 1D array
                # dlsim or lfilter. lfilter is standard.
                from scipy.signal import lfilter
                # simple filter
                y_f = lfilter(b, a, signal[ch])
                filtered.append(y_f)
            signal = np.array(filtered)
        
        # Convert to tensor
        signal_tensor = torch.tensor(signal, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        domain_tensor = torch.tensor(domain, dtype=torch.long)

        # Return (x, y, d)
        return signal_tensor, label_tensor, domain_tensor

    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()
