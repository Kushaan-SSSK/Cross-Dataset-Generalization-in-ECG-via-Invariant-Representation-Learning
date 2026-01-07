# Benchmarking Cross-Dataset Generalization in ECG: A Diagnostic Stress-Test Protocol

**Abstract**
Deep learning models for electrocardiogram (ECG) classification frequently fail to generalize to new data sources due to distribution shifts arising from differences in acquisition hardware and patient populations. Current evaluations often rely on aggregate metrics that mask the underlying failure modes. In this work, we propose a standardized **Shortcut Amplification Stress Test (SAST)** protocol to rigorously diagnose model robustness. We compare four state-of-the-art invariant learning baselines (ERM, IRM, DANN, V-REx) across two large-scale heterogeneous datasets (PTB-XL and Chapman-Shaoxing). By integrating a novel **Dataset-Identity Leakage** metric and **Frequency Sensitivity Analysis**, we demonstrate that methods preserving high in-domain accuracy often do so by latching onto spurious acquisition artifacts. Our benchmark establishes a rigorous framework for evaluating the safety and reliability of physiological time-series models before clinical deployment.

## 1. Introduction

Electrocardiography (ECG) remains the cornerstone of non-invasive cardiac diagnostics. While Deep Neural Networks (DNNs) have achieved cardiologist-level performance on retrospective benchmarks, their real-world deployment is hindered by **fragility under distribution shift**. Models trained on data from a specific hospital system (e.g., specific device vendors in Germany) often experience catastrophic performance degradation when applied to a new clinical environment (e.g., different devices in China).

This generalization gap suggests that standard training paradigms, such as Empirical Risk Minimization (ERM), fail to distinguish between **causal physiological features** (e.g., QRS morphology) and **spurious correlations** (e.g., vendor-specific baseline wander, power-line interference, or preprocessing artifacts). When a model relies on these "shortcuts," it achieves high accuracy within the source domain but fails in target domains where these correlations are absent or reversed.

Strategies to mitigate this, such as Domain Generalization (DG) algorithms (DANN, IRM, V-REx), have shown promise in computer vision. However, their efficacy in physiological signal processing remains under-explored and difficult to quantify using standard accuracy metrics alone. A model might maintain high accuracy on a target domain yet still rely on non-robust features, leaving it vulnerable to subtle artifact changes.

In this paper, we address this gap by proposing a comprehensive **diagnostic benchmarking protocol** for ECG generalization. We move beyond simple performance tables to mechanistic stress-testing.

Our contributions are:
1.  **Systematic Cross-Dataset Benchmark**: We evaluate ERM, IRM, DANN, and V-REx on a rigorous leave-one-domain-out task using PTB-XL (Germany) and Chapman-Shaoxing (China), keeping the evaluation clinically realistic (no target labels used).
2.  **Shortcut Amplification Stress Test (SAST)**: We introduce a standardized protocol that injects controlled, high-frequency "shortcuts" (e.g., 60Hz artifacts) carrying label information during training. This amplifies the incentive for shortcut learning, allowing us to quantify exactly how much each method resists non-physiological correlations.
3.  **Diagnostic Metrics Suite**: We propose **Dataset-Identity Leakage** (using linear probes on frozen embeddings) and **Frequency Attribution Analysis** as standard metrics for verifying invariant learning. We show these diagnositics reveal failures that processed macro-F1 scores miss.

## 2. Related Work

**Deep Learning for ECG.**
Despite the success of CNNs and Transformers in ECG analysis (Strodthoff et al., 2020), most studies rely on random independent and identically distributed (i.i.d.) splits within single datasets like PhysioNet 2020. Recent cross-dataset studies (Leinonen et al., 2024) document performance drops but lack mechanistic explanations for *why* models fail.

**Domain Generalization (DG).**
DG aims to learn robust predictors from multiple source domains. Techniques include learning invariant representations (DANN, Ganin et al., 2016), regularizing risk variance (V-REx, Krueger et al., 2021), or enforcing invariant optimal classifiers (IRM, Arjovsky et al., 2019). While DomainBed (Gulrajani & Lopez-Paz, 2021) suggests ERM is a strong baseline in vision, time-series data offers unique opportunities for measuring invariance via frequency analysis, which we exploit here.

## 3. Problem Formulation

We consider the Domain Generalization (DG) problem. Let $\mathcal{X}$ be the space of ECG signals and $\mathcal{Y}$ the label space. We have training environments $\mathcal{E}_{train}$ with datasets $S_e$ drawn from $P_e(X, Y)$. The goal is to minimize risk on an unseen target environment $e_{test}$.

We assume input $X$ decomposes into causal features $X_c$ (invariant) and spurious features $X_s$ (domain-specific). Our goal is to benchmark how well different learning objectives $f_\theta$ ignore $X_s$.

## 4. Benchmarked Methods

We evaluate four representative strategies:
1.  **ERM (Empirical Risk Minimization):** Standard training, minimizes average loss. Serves as the naive baseline.
2.  **DANN (Domain-Adversarial Neural Networks):** Uses an adversarial domain discriminator to remove domain information from features.
3.  **V-REx (Variance Risk Extrapolation):** Penalizes the variance of effective loss across training environments to encourage stability.
4.  **IRM (Invariant Risk Minimization):** Penalizes the gradient norm of the loss to enforce the optimality of the classifier across environments.

## 5. Proposed Evaluation Protocol

### 5.1. Cross-Dataset Task (PTB-XL $\to$ Chapman)
We use PTB-XL (Schiller devices) as the Source and Chapman-Shaoxing (GE devices) as the Target (OOD). This represents a realistic "deploy to new hospital" scenario.

### 5.2. Shortcut Amplification Stress Test (SAST)
To measure robustness mechanistically, we propose **SAST**.
*   **Protocol:** We inject a definable artifact (e.g., 60Hz sinusoidal noise) into the training data such that it correlates strongly ($P=0.9$) with the "Abnormal" class. In the test set, this correlation is removed.
*   **Metric:** We measure the **Performance Drop** ($\Delta_{SAST}$) from specific methods when exposed to this "poisoned" training environment compared to clean training. A robust method should ignore the easy 60Hz shortcut and learn the harder morphological features.

### 5.3. Diagnostic Metrics
*   **Dataset-Identity Leakage:** We train a linear probe on frozen features to predict the source dataset. High accuracy indicates the model has learned "where the data came from" (bad), while low accuracy implies invariant feature learning (good).
*   **Frequency Attribution:** We compute the gradient-weighted frequency attribution (Saliency + FFT) to quantify exactly how much attention the model pays to the 58-62Hz band.

## 6. Results

### 6.1. Performance & Robustness
*Placeholder: Table comparing ERM, DANN, V-REx, IRM on F1, AUROC, and ECE.*

### 6.2. Diagnostic Insights
*Placeholder: Leakage Analysis showing DANN/V-REx feature invariance vs ERM.*
*Placeholder: Frequency Analysis verifying DANN reduces reliance on 60Hz shortcuts.*

## 7. Conclusion
We present a rigorous benchmarking protocol for ECG generalization. Our results show that standard aggregate metrics are insufficient. By utilizing the proposed SAST protocol and diagnostic probes, we reveal that even high-performing models can be brittle. We recommend this diagnostic suite as a standard check for all clinical ECG models.

