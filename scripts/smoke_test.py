
import sys
import os
import torch
import hydra
import omegaconf
from packaging import version
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.models.resnet1d import ResNet1d
    from scripts.run_embc_fix import resolve_data_path
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def check_versions():
    print(f"Hydra version: {hydra.__version__}")
    print(f"OmegaConf version: {omegaconf.__version__}")
    
    # Simple check, not strictly failing if it's slightly off unless critical
    # Requirements said >= 1.3.2 and >= 2.3.0
    h_ver = version.parse(hydra.__version__)
    o_ver = version.parse(omegaconf.__version__)
    
    if h_ver < version.parse("1.3.2"):
        print("WARNING: hydra-core version is older than 1.3.2")
    if o_ver < version.parse("2.3.0"):
        print("WARNING: omegaconf version is older than 2.3.0")

def test_resnet_feats():
    print("\nTesting ResNet1d return_feats...")
    model = ResNet1d(input_channels=12, num_classes=5)
    model.eval()
    x = torch.randn(2, 12, 1000)
    
    # 1. Normal call
    out = model(x)
    assert isinstance(out, torch.Tensor), "Default output should be tensor"
    print("Normal forward pass: OK")
    
    # 2. return_feats=True
    out2 = model(x, return_feats=True)
    assert isinstance(out2, tuple), "Output with return_feats should be tuple"
    assert len(out2) == 2, "Tuple length should be 2"
    logits, feats = out2
    assert logits.shape == out.shape, "Logits shape mismatch"
    assert feats.ndim == 2, "Feats should be 2D"
    assert feats.shape[0] == 2, "Batch size mismatch in feats"
    print(f"return_feats pass: OK. Feats shape: {feats.shape}")

def test_path_resolution():
    print("\nTesting path resolution...")
    # create dummy files
    p_primary = "dummy_primary.txt"
    p_fallback = "dummy_fallback.txt"
    
    with open(p_primary, "w") as f: f.write("test")
    
    # Case 1: Primary exists
    res = resolve_data_path(p_primary, p_fallback)
    assert res == p_primary, f"Should return primary, got {res}"
    print("Case 1 (Primary exists): OK")
    
    # Case 2: Primary missing, Fallback exists
    os.remove(p_primary)
    with open(p_fallback, "w") as f: f.write("test")
    res = resolve_data_path(p_primary, p_fallback)
    # resolve_data_path converts to string path, might be absolute or relative depending on input?
    # Actually it wraps in Path(str) and returns str(path). 
    # If input was relative, Path might verify existence relative to CWD.
    assert Path(res).name == p_fallback, f"Should return fallback, got {res}"
    print("Case 2 (Fallback exists): OK")
    
    # Cleanup
    if os.path.exists(p_fallback): os.remove(p_fallback)

if __name__ == "__main__":
    check_versions()
    test_resnet_feats()
    test_path_resolution()
    print("\nSmoke Test Passed!")
