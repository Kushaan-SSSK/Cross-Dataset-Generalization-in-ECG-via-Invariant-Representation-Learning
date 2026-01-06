import hydra
from omegaconf import DictConfig
import logging
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import os
import importlib
import inspect
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from src.dataset import ECGDataset

log = logging.getLogger(__name__)

def forward_logits(model, x):
    """
    Normalize forward outputs to logits tensor.
    Supports:
      - logits
      - (logits, extra...)
      - dict with logits-like key
    """
    out = model(x)
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
        return out[0]
    if isinstance(out, dict):
        for k in ["logits", "y_hat", "pred", "output"]:
            if k in out and isinstance(out[k], torch.Tensor):
                return out[k]
    raise RuntimeError(f"Cannot extract logits from model output type={type(out)}")

def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_targets, all_domains = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            batch = [b.to(device) for b in batch]
            x, y, d = batch[:3]
            logits = forward_logits(model, x)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_domains.append(d.cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets), np.concatenate(all_domains)

def _get_ids_from_splits(splits: dict, split_name: str):
    ids = splits.get(split_name, [])
    if ids:
        return ids
    out = []
    for k, v in splits.items():
        if split_name in str(k).lower():
            out.extend(v)
    return out

def _pick_method_class(module, run_name: str):
    """
    Pick the most likely wrapper class from a module by inspecting classes defined there.
    """
    run = run_name.lower()

    candidates = []
    for name, obj in vars(module).items():
        if not inspect.isclass(obj):
            continue
        # Keep classes defined in this module
        if obj.__module__ != module.__name__:
            continue
        # Must be a torch nn.Module (your methods should be)
        try:
            if not issubclass(obj, torch.nn.Module):
                continue
        except Exception:
            continue
        candidates.append(obj)

    if not candidates:
        raise RuntimeError(f"No torch.nn.Module classes found in {module.__name__}")

    # Prefer class whose name matches the run keyword
    def score(cls):
        n = cls.__name__.lower()
        s = 0
        if "dann" in run and "dann" in n: s += 5
        if "vrex" in run and "vrex" in n: s += 5
        if "irm" in run and "irm" in n: s += 5
        if ("v2" in run or "disentangled" in run) and ("disentangled" in n or "pid" in n or "v2" in n): s += 5
        if "erm" in run and "erm" in n: s += 5
        # Slightly prefer simpler names
        s -= (len(n) / 100.0)
        return s

    candidates = sorted(candidates, key=score, reverse=True)
    return candidates[0]

def build_method_from_run_name(run_name: str, backbone, num_classes: int):
    rn = run_name.lower()
    if "dann" in rn:
        mod = "dann"
    elif "vrex" in rn:
        mod = "vrex"
    elif "irm" in rn:
        mod = "irm"
    elif rn == "v2" or "disentangled" in rn:
        mod = "disentangled"
    else:
        mod = "erm"

    module = importlib.import_module(f"src.methods.{mod}")
    cls = _pick_method_class(module, run_name)
    try:
        return cls(backbone, num_classes)
    except TypeError:
        # In case signature differs, give a clearer error
        raise RuntimeError(f"Method class {cls.__name__} in {module.__name__} does not accept (backbone, num_classes).")

@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def main(cfg: DictConfig):
    log.info("Generating Results...")

    split_name = cfg.get("eval", {}).get("split", "val")
    base_dir   = cfg.get("eval", {}).get("base_dir", "outputs/baselines")
    methods    = cfg.get("eval", {}).get("methods", ["erm_fixed", "dann_fixed", "vrex_fixed", "irm_fixed", "v2"])
    out_dir    = cfg.get("eval", {}).get("out_dir", "results")

    log.info(f"Eval split: {split_name}")
    log.info(f"Base dir: {base_dir}")
    log.info(f"Runs: {methods}")

    manifest_df = pd.read_csv(cfg.data.paths.manifest_path)
    with open(cfg.data.paths.split_path, "r") as f:
        splits = json.load(f)

    ids = _get_ids_from_splits(splits, split_name)
    if not ids:
        raise RuntimeError(f"No IDs found for split='{split_name}' in {cfg.data.paths.split_path}")

    df = manifest_df[manifest_df["unique_id"].isin(ids)]
    df = df[df["dataset_source"].isin(["ptbxl", "chapman"])]

    import h5py
    with h5py.File(cfg.data.paths.processed_path, "r") as f:
        existing = set(f.keys())
    df = df[df["unique_id"].isin(existing)].reset_index(drop=True)

    ds = ECGDataset(df, cfg.data.paths.processed_path)
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    domain_map = {0: "PTB-XL", 1: "Chapman"}
    results = []

    for run_name in methods:
        ckpt_path = os.path.join(base_dir, run_name, "best_model.pt")
        if not os.path.exists(ckpt_path):
            log.warning(f"Checkpoint not found for {run_name} at {ckpt_path}. Skipping.")
            continue

        log.info(f"Evaluating {run_name} from {ckpt_path}...")

        backbone = hydra.utils.instantiate(cfg.model)
        model = build_method_from_run_name(run_name, backbone, cfg.model.num_classes)

        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)

        preds, targets, domains = evaluate_model(model, loader, device)

        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average="macro", zero_division=0)
        results.append({"run": run_name, "split": split_name, "domain": "Overall", "acc": acc, "f1": f1})

        for d_idx, d_name in domain_map.items():
            mask = (domains == d_idx)
            if mask.sum() > 0:
                d_acc = accuracy_score(targets[mask], preds[mask])
                d_f1 = f1_score(targets[mask], preds[mask], average="macro", zero_division=0)
                results.append({"run": run_name, "split": split_name, "domain": d_name, "acc": d_acc, "f1": d_f1})

        cm = confusion_matrix(targets, preds, normalize="true")
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues")
        plt.title(f"Confusion Matrix ({split_name}): {run_name}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.savefig(os.path.join(out_dir, f"cm_{split_name}_{run_name}.png"))
        plt.close()

    if not results:
        log.warning("No models evaluated.")
        return

    df_res = pd.DataFrame(results)
    out_csv = os.path.join(out_dir, f"final_metrics_{split_name}.csv")
    df_res.to_csv(out_csv, index=False)

    print("\n=== Final Results ===")
    print(df_res)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_res, x="run", y="f1", hue="domain")
    plt.title(f"Method Comparison by Domain ({split_name})")
    plt.ylabel("Macro F1 Score")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"method_comparison_{split_name}.png"))
    plt.close()

    log.info(f"Saved metrics to {out_csv} and plots to {out_dir}/")

if __name__ == "__main__":
    main()
