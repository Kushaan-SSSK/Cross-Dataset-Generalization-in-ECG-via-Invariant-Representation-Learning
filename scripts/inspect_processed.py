
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import random

def inspect_hdf5(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            keys = list(f.keys())
            print(f"Total records in HDF5: {len(keys)}")
            
            if not keys:
                print("No records found.")
                return

            # Sample checks
            sample_keys = random.sample(keys, min(5, len(keys)))
            print("\n--- Sample Records ---")
            for key in sample_keys:
                data = f[key][()]
                print(f"Key: {key}, Shape: {data.shape}, Mean: {data.mean():.4f}, Std: {data.std():.4f}")
                
            # Verification Logic
            # Check for shape consistency (most should be 1000, 12)
            # Note: MIT-BIH might be different if we didn't crop it (which we handled in the script by skipping crop_or_pad for 'mitbih'?)
            # Wait, my script logic was:
            # if source in ['ptbxl', 'chapman']: crop_or_pad
            # else (mitbih): leave as is.
            # So MIT-BIH shapes will be variable and long.
            
            print("\n--- Shape Distribution Probe (first 100) ---")
            shapes = []
            for k in keys[:100]:
                shapes.append(f[k].shape)
            
            from collections import Counter
            print(Counter(shapes))
            
            # Plot one
            key = sample_keys[0]
            data = f[key][()]
            plt.figure(figsize=(10, 4))
            plt.plot(data)
            plt.title(f"Processed Signal: {key} {data.shape}")
            plt.tight_layout()
            plt.savefig("sample_processed_signal.png")
            print(f"\nSaved plot to sample_processed_signal.png")

    except OSError as e:
        print(f"Could not open {file_path}. It might be locked by the writer process or does not exist.\nError: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to .h5 file")
    args = parser.parse_args()
    
    inspect_hdf5(args.path)
