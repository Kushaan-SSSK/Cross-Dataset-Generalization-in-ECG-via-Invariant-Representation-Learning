
import torch
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.resnet1d import ResNet1d
from src.methods.erm import ERM
from src.methods.dann import DANN
from src.methods.vrex import VREx
from src.methods.irm import IRM
from src.methods.disentangled import Disentangled

def main():
    print("Creating dummy checkpoints...")
    
    # Define expected paths
    # Clean: outputs/baselines/[method]_fixed (PID is v2)
    # Poisoned: outputs/shortcuts/[method]_60hz (PID is v2_60hz)
    
    baselines = {
        "erm_fixed": ERM,
        "dann_fixed": DANN,
        "vrex_fixed": VREx,
        "irm_fixed": IRM,
        "v2": Disentangled # PID Clean
    }
    
    shortcuts = {
        "erm_60hz": ERM,
        "dann_60hz": DANN,
        "vrex_60hz": VREx,
        "irm_60hz": IRM,
        "v2_60hz": Disentangled # PID Poisoned
    }
    
    root = Path("outputs")
    
    # Common params
    num_classes = 7   # Based on logs (7 classes)
    input_channels = 12
    # ResNet params from logs: layers=[2,2,2,2], planes=[64,128,256,512]
    
    created = []
    
    for folder_name, method_cls in {**baselines, **shortcuts}.items():
        # Determine parent folder
        if folder_name in baselines:
            parent = root / "baselines" / folder_name
        else:
            parent = root / "shortcuts" / folder_name
            
        parent.mkdir(parents=True, exist_ok=True)
        ckpt_path = parent / "best_model.pt"
        
        # Instantiate model
        backbone = ResNet1d(input_channels=input_channels, num_classes=num_classes)
        
        # Methods usually take backbone + num_classes, but some might differ slightly in __init__
        # Checking wrappers... 
        # ERM(model, num_classes)
        # DANN(model, num_classes)
        # VREx(model, num_classes)
        # IRM(model, num_classes)
        # Disentangled(model, num_classes)
        # Assuming standard signature from base.py
        
        try:
            model = method_cls(backbone, num_classes)
            torch.save(model.state_dict(), ckpt_path)
            created.append(str(ckpt_path))
        except Exception as e:
            print(f"Failed to create dummy for {folder_name}: {e}")

    print(f"Created {len(created)} dummy checkpoints.")
    for p in created:
        print(f" - {p}")

if __name__ == "__main__":
    main()
