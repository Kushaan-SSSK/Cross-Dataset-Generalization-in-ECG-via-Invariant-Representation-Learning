# Cross-Dataset Generalization in ECG via Invariant Representation Learning

## 1. Abstract
Deep learning models for ECG diagnosis often suffer from performance degradation when deployed to hospitals different from their training data (Out-of-Distribution shift). This project investigates **Invariant Representation Learning** techniques to mitigate this issue. We benchmark standard Empress Risk Minimization (ERM) against domain-invariant methods—Domain-Adversarial Neural Networks (DANN) and Variance Risk Extrapolation (V-REx)—on a multi-source dataset comprising PTB-XL and Chapman-Shaoxing. We further propose a Disentangled Representation Learning framework (V2) to explicitly separate disparate hospital-specific features from stable biological features.

## 2. Methodology
We implement and compare four distinct training paradigms:

### 2.1. Empirical Risk Minimization (ERM)
The standard baseline. Minimizes the average cross-entropy loss across all data, effectively pooling all domains.
$$ \min_{\theta} \mathbb{E}_{(x,y) \sim P_{train}} [\ell(f_\theta(x), y)] $$
*   **Hypothesis**: High in-distribution accuracy, but susceptible to learning spurious correlations (low robustness).

### 2.2. Domain-Adversarial Neural Networks (DANN)
A minimax game where a feature extractor tries to confuse a domain discriminator while maintaining diagnostic accuracy.
*   **Loss**: $\mathcal{L}_{task} - \alpha \mathcal{L}_{domain}$
*   **Goal**: Ensure features $g(x)$ contain no information about the hospital source.

### 2.3. Variance Risk Extrapolation (V-REx)
Enforces robustness by penalizing the *variance* of the loss across training domains.
*   **Loss**: $\mathcal{L}_{avg} + \beta \text{Var}(\{\mathcal{L}_e\}_{e \in \mathcal{E}})$
*   **Goal**: Force the model to be equally performant across all hospitals, preventing it from overfitting to "easy" domains.

### 2.4. Proposed Method (Disentangled V2)
An encoder-decoder architecture that splits the latent space into $Z_{content}$ (disease features) and $Z_{style}$ (hospital features), enforcing orthogonality to ensure $Z_{content}$ is domain-invariant.

## 3. Experimental Setup

### 3.1. Datasets (Two-Source Domain)
We utilize a rigorous multi-source setting combining two major open-access databases:
*   **PTB-XL** (Germany): High-quality 12-lead ECGs from the Physikalisch-Technische Bundesanstalt.
*   **Chapman-Shaoxing** (China): 12-lead ECGs from Chapman University and Shaoxing People's Hospital.

**Exclusions**:
*   **MIT-BIH Arrhythmia Database**: While considered, this dataset was excluded from the 12-lead training pipeline because it consists of 2-lead ambulatory recordings, which are dimensionally incompatible with the 12-lead architecture (ResNet1d-18).
*   **Ningbo First Hospital**: Often paired with Chapman, it was not included in this iteration to maintain a balanced 2-domain shift (Europe vs. Asia).

**Preprocessing**:
*   Resampled to 250Hz.
*   Bandpass Filter (0.5Hz - 50Hz).
*   Z-Score Normalization.
*   Task: Multi-label classification (7 harmonized classes based on SNOMED-CT codes).

### 3.2. Model Architecture
*   **Backbone**: ResNet1d-18 (Optimization of He et al. for 1D time-series).
*   **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-2).
*   **Scheduler**: CosineAnnealingWarmRestarts.

## 4. Execution & Reproducibility

### 4.1. Environment Setup
```bash
pip install -r requirements.txt
```

### 4.2. Training the Benchmarks
To reproduce Table 1 (Baselines):
```bash
python scripts/run_baselines.ps1
```
This executes:
1.  ERM (50 Epochs)
2.  DANN (50 Epochs, alpha=1.0)
3.  V-REx (50 Epochs, beta=10.0)

### 4.3. Stress Testing (Robustness Analysis)
To generate the robustness curves (Figure 4):
```bash
python scripts/run_stress_test.py
```
This injects increasing levels of baseline wander and Gaussian noise to measure how quickly performance degrades.

## 5. Results & Metrics (Preliminary)
*   **ERM**: Achieves high validation accuracy (~90%) but shows signs of overfitting/memorization (Train Acc >98%).
*   **DANN/V-REx**: Currently achieving comparable F1 scores (~82-84%). Stability analysis pending final stress tests.

## 6. Project Structure
*   `src/models`: ResNet1d implementations.
*   `src/methods`: Trainer logic for ERM, DANN, V-REx, Disentangled.
*   `src/dataset.py`: Harmonized dataloader handling HDF5 signal loading.
*   `config/`: Hydra configuration files.
*   `scripts/`: Analysis and plotting utilities.


## 7. Technical Codebase Reference

### 7.1. Core Pipeline (`src/`)

#### `src/train.py`
The central training orchestration script.
*   **Hydra Configuration**: Uses `@hydra.main` to dynamically load configurations from `config/`.
*   **Method Factory**: Instantiates the correct algorithm (ERM/DANN/V-REx) based on `cfg.method._target_`.
*   **Data Handling**: Loads `master_manifest.csv`, filters out incompatible sources (MIT-BIH), and creates `ECGDataset` instances.
*   **Training Loop**: Implements the epoch-based loop, handles `batch` unpacking (which can be `(x, y)` or `(x, y, d)`), and manages `best_model.pt` checkpointing based on Validation F1 validation.

#### `src/dataset.py`
Custom PyTorch Dataset `ECGDataset`.
*   **HDF5 Integration**: Efficiently reads signals from `data/processed/signals.h5` using `unique_id` lookups.
*   **Label Mapping**: Converts textual SNOMED codes into multi-hot encoded tensors based on the mapping defined in `config/data/default.yaml`.
*   **Domain Handling**: Returns a 3-tuple `(signal, label, domain_id)` where `domain_id` maps `ptbxl->0`, `chapman->1`. This is crucial for DANN/V-REx.

#### `src/models/resnet1d.py`
A 1D Adaptation of the ResNet-18 architecture.
*   **Conv1d Layers**: Replaces standard 2D convolutions with 1D convolutions (kernel size 7 for the first layer, 3 for subsequent blocks) to capture temporal ECG dynamics.
*   **Adaptive Pooling**: Uses `AdaptiveAvgPool1d` to handle variable length signals (though we fix length to 2500 for batching).

### 7.2. Methods (`src/methods/`)

#### `src/methods/erm.py`
*   **Class**: `ERM`
*   **Logic**: Standard `CrossEntropyLoss`. The baseline "lower bound" of robustness.

#### `src/methods/dann.py`
*   **Class**: `DANN`
*   **Technique**: **Gradient Reversal Layer (GRL)**.
*   **Logic**: Forward pass calculates `TaskLoss + Alpha * DomainLoss`. Backward pass *flips* the sign of the gradient from the Domain Classifier before it reaches the Feature Extractor. This forces the Feature Extractor to maximize Domain Loss (confuse the discriminator).

#### `src/methods/vrex.py`
*   **Class**: `VREx`
*   **Technique**: Variance Penalty.
*   **Logic**: Calculates loss *per domain* independently. Adds `Beta * Variance(DomainLosses)` to the total loss. This penalizes the model if it performs well on one hospital but poorly on another.

#### `src/methods/disentangled.py` (Proposed V2)
*   **Technique**: Orthogonal Latent Space.
*   **Structure**: Splits the bottleneck into `Z_content` (Disease) and `Z_style` (Hospital). Explicitly penalizes mutual information between them.

### 7.3. Analysis Scripts (`scripts/`)

#### `scripts/01_build_manifest.py`
*   **Role**: Data harmonization.
*   **Logic**: Parses raw header files (`.hea`) from PTB-XL/Chapman, normalizes filenames, harmonizes diagnostic labels to SNOMED-CT, and produces `master_manifest.csv`.

#### `scripts/02_preprocess_signals.py`
*   **Role**: Signal Processing.
*   **Logic**: Loads raw waveforms (via `wfdb`), resamples to 250Hz, applies 0.5-50Hz Bandpass filter, performs Z-score normalization, and saves as a compressed HDF5 archive.

#### `scripts/run_stress_test.py`
*   **Role**: Robustness Verification.
*   **Technique**: Loads a trained model and applies `src/utils/perturbations.py` (Gaussian Noise, Baseline Wander) at increasing severities (0.0 to 1.0).
*   **Output**: Generates CSVs of F1 score vs. Noise Leve, used for Figure 4.

#### `scripts/generate_results.py`
*   **Role**: Final Artifact Generation.
*   **Logic**: Computes Confusion Matrices, Per-Class F1 scores, and aggregates results into Table 1 for the paper.

----
*Authored by Antigravity*
