
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset
import numpy as np

from omegaconf import OmegaConf
class ECGDataset(Dataset):
    """
    ECG Dataset reading from HDF5 processed file.
    """
    def __init__(self, manifest_df, hdf5_path, task_label_col='task_a_label', shortcut_cfg=None, split='train', binary_labels=False):
        """
        Args:
            manifest_df (pd.DataFrame): DataFrame containing 'unique_id' and labels.
            hdf5_path (str): Path to the processed .h5 file.
            task_label_col (str): Column name for the target label.
            shortcut_cfg (DictConfig): Configuration for shortcut injection.
            split (str): 'train', 'val', or 'test'.
            binary_labels (bool): If True, maps 0->0 (Normal) and >0->1 (Abnormal).
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
        self.binary_labels = binary_labels
        self.h5_file = None

    def __len__(self):
        return len(self.manifest_df)
        
    def _add_shortcut(self, signal):
        """Add specified artifact noise (Mains, BW, EMG)."""
        if self.shortcut_cfg is None: return signal
        
        sType = getattr(self.shortcut_cfg, 'type', 'mains')
        amp = OmegaConf.select(self.shortcut_cfg, "amplitude", default=0.1)
        fs = 100 # Assumed
        t = np.arange(signal.shape[0]) / fs
        
        noise = np.zeros_like(t)
        
        if sType == 'mains':
            freq = getattr(self.shortcut_cfg, 'freq', 60)
            noise = amp * np.sin(2 * np.pi * freq * t)
            
        elif sType == 'bw': # Baseline Wander
            # Low freq sinusoid (0.5Hz) + Linear drift
            noise = amp * np.sin(2 * np.pi * 0.5 * t) + (amp * 0.5 * t)
            
        elif sType == 'emg': # Electromyographic (Muscle) Noise
            # High freq Gaussian noise burst
            # Burst covers random 50% of the signal
            rng = np.random.default_rng(seed=int(signal[0,0]*1000)) 
            mask = rng.random(len(t)) > 0.5
            raw_noise = rng.normal(0, 1, len(t))
            # EMG has high freq content
            noise = amp * raw_noise * mask
            
        # Add to Lead I (index 0)
        signal[:, 0] += noise
        return signal

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.hdf5_path, 'r')

        row = self.manifest_df.iloc[idx]
        unique_id = row['unique_id']
        label = row[self.task_label_col]
        
        # Binary Mapping Logic (Explicit 0 vs Rest)
        # 0 = Normal, 1 = Abnormal
        if self.binary_labels:
            label = 0 if label == 0 else 1
        
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
            if getattr(self.shortcut_cfg, 'force', False):
                inject = True
            elif self.split == 'train':
                p_corr = self.shortcut_cfg.correlation
                
                # Check target label (Assuming Binary: 0=Normal, 1=Abnormal)
                # If binary_labels=True, label is 0 or 1.
                # If binary_labels=False, raw classes > 0 are abnormal.
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
