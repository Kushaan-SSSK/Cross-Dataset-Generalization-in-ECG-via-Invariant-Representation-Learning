# Collaboration Notes

## Current Status (2026-01-04)
The codebase has been refactored for the ICML submission.
1.  **MIT-BIH Removed:** The code is now strictly for PTB-XL and Chapman-Shaoxing.
2.  **Synthetic Shortcut Logic Implemented:** `src/dataset.py` now supports on-the-fly 60Hz noise injection.

## Instructions for Collaborator

### 1. Setup
```bash
git pull origin main
pip install -r requirements.txt
```

### 2. Run Standard Experiments (Table 1)
Run these to get the baseline and method performance on clean data. **These should run smoothly.**
```bash
# ERM
python -m src.train method=erm train.epochs=50 hydra.job.chdir=True hydra.run.dir=outputs/baselines/erm_fixed

# DANN
python -m src.train method=dann train.epochs=50 hydra.job.chdir=True hydra.run.dir=outputs/baselines/dann_fixed

# V-REx
python -m src.train method=vrex train.epochs=50 hydra.job.chdir=True hydra.run.dir=outputs/baselines/vrex_fixed

# IRM (Invariant Risk Minimization)
python -m src.train method=irm train.epochs=50 hydra.job.chdir=True hydra.run.dir=outputs/baselines/irm_fixed

# Disentangled V2 (Proposed)
python -m src.train method=disentangled train.epochs=50 hydra.job.chdir=True hydra.run.dir=outputs/baselines/v2
```

### 3. Run Synthetic Shortcut Experiments (Figure 2)
Run these recursively to generate data for the "Shortcut Sensitivity" plot. This enables 60Hz noise injection during training (creating "poisoned" models).
```bash
# ERM on Poisoned Data
python -m src.train method=erm data.shortcut.use_shortcut=true experiment_name=erm_shortcut hydra.run.dir=outputs/shortcuts/erm_60hz

# DANN on Poisoned Data
python -m src.train method=dann data.shortcut.use_shortcut=true experiment_name=dann_shortcut hydra.run.dir=outputs/shortcuts/dann_60hz

# V-REx on Poisoned Data
python -m src.train method=vrex data.shortcut.use_shortcut=true experiment_name=vrex_shortcut hydra.run.dir=outputs/shortcuts/vrex_60hz

# IRM on Poisoned Data
python -m src.train method=irm data.shortcut.use_shortcut=true experiment_name=irm_shortcut hydra.run.dir=outputs/shortcuts/irm_60hz

# Disentangled on Poisoned Data
python -m src.train method=disentangled data.shortcut.use_shortcut=true experiment_name=disentangled_shortcut hydra.run.dir=outputs/shortcuts/v2_60hz
```

### 4. Evaluation
After training, verify results using `scripts/run_stress_test.py` or check the logs in `outputs/`.
