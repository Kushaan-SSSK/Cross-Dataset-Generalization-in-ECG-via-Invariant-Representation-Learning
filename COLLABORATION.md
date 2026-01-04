
# GPU Handoff Instructions
Send this file to your colleague. It assumes they have cloned the repo.

## 1. Environment Setup
```bash
pip install -r requirements.txt
```

## 2. Data Preparation (If not already done)
If they have the raw data folders (`ptb-xl...`, `chapman...`, `mit-bih...`) in the root:
```bash
# 1. Build the manifest (Harmonize labels)
python scripts/01_build_manifest.py

# 2. Preprocess Signals (Resample, Filter, Save to HDF5)
python scripts/02_preprocess_signals.py
```

## 3. Training Commands (Run in Order)
These commands render the models to specific directories. 
**Note**: `hydra.job.chdir=True` is CRITICAL to prevent overwriting.

### A. Train Baseline: ERM (Standard) / 50 Epochs
```bash
python -m src.train method=erm train.epochs=50 hydra.job.chdir=True hydra.run.dir=outputs/baselines/erm_fixed
```

### B. Train Baseline: DANN (Robust) / 50 Epochs
```bash
python -m src.train method=dann train.epochs=50 hydra.job.chdir=True hydra.run.dir=outputs/baselines/dann_fixed
```

### C. Train Proposed Method: V2 (Disentangled) / 50 Epochs
```bash
python -m src.train method=disentangled train.epochs=50 hydra.job.chdir=True hydra.run.dir=outputs/baselines/v2
```

## 4. Verification & Stress Testing
Once training is done, run these to generate the paper's figures.

### A. Stress Test ERM
```bash
python scripts/run_stress_test.py +checkpoint_path="outputs/baselines/erm_fixed/best_model.pt" +result_name="results_erm.csv"
```

### B. Stress Test DANN
```bash
python scripts/run_stress_test.py +checkpoint_path="outputs/baselines/dann_fixed/best_model.pt" +result_name="results_dann.csv"
```

### C. Stress Test V2 (Proposed)
```bash
python scripts/run_stress_test.py +checkpoint_path="outputs/baselines/v2/best_model.pt" +result_name="results_v2.csv"
```

### D. Stress Test V-REx (If they need to re-run it)
```bash
# Training (Optional if you have the file)
python -m src.train method=vrex train.epochs=50 hydra.job.chdir=True hydra.run.dir=outputs/baselines/vrex_fixed

# Stress Test
python scripts/run_stress_test.py +checkpoint_path="outputs/baselines/vrex_fixed/best_model.pt" +result_name="results_vrex.csv"
```

## 5. Generate Final Results
This aggregates all CSVs and makes the plots.
```bash
python scripts/generate_results.py
```
