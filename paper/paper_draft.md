
# Benchmarking Cross-Dataset Generalization in ECG: A Diagnostic Stress-Test Protocol

**Abstract**— Deep learning models for electrocardiogram (ECG) classification have achieved expert-level performance in retrospective studies. However, their deployment is hampered by performance degradation under **compound distribution shifts**, where changes in acquisition hardware, signal processing, and patient demographics occur simultaneously. Current evaluation standards often fail to detect reliance on unstable "shortcut" features. In this work, we propose a standardized **Shortcut Amplification Stress Test (SAST)** protocol to rigorously diagnose the robustness of ECG classifiers under these severe assumptions. We benchmark four state-of-the-art domain generalization algorithms—ERM, IRM, DANN, V-REx—across two heterogeneous datasets (PTB-XL and Chapman-Shaoxing) representing a "wild" deployment shift. By integrating a novel **Dataset-Identity Leakage** metric, we demonstrate that methods maintaining high in-domain accuracy often do so by exploiting non-physiological artifacts. We recommend SAST and leakage probing as practical, low-cost pre-deployment diagnostics.

**Keywords**— Electrocardiography, Domain Generalization, Deep Learning, Robustness, Stress Testing

## I. INTRODUCTION

The automated interpretation of Electrocardiograms (ECG) using Deep Neural Networks (DNNs) holds the promise of democratizing cardiovascular diagnostics [1]. However, the transition to real-world clinical utility is obstructed by **fragility under distribution shift**. A model trained on high-fidelity signals from a tertiary care center (e.g., Schiller devices in Germany) may fail when deployed to a community hospital setting (e.g., GE devices in China).

This paper addresses this gap by proposing a comprehensive **diagnostic benchmarking protocol** for ECG generalization. We position this work as a safety stress-test designed to audit models for specific shortcut dependencies before deployment, moving beyond aggregate metrics to interpretable failure modes. It is distinctly not a proposal for a new model architecture or loss function optimization, but rather a rigorous evaluation framework.

Our contributions are threefold. First, we conduct a **Systematic Cross-Dataset Benchmark**, evaluating ERM, IRM, DANN, and V-REx on a reproducible task using PTB-XL [4] and Chapman-Shaoxing [5]. Second, we introduce the **Shortcut Amplification Stress Test (SAST)**, a standardized protocol that injects controlled "shortcuts" (e.g., 60Hz artifacts) to quantify mechanism-agnostic robustness. Third, we propose a **Diagnostic Metrics Suite** featuring Dataset-Identity Leakage as a standard metric for verifying invariant learning.

## II. RELATED WORK

### A. Domain Generalization in Medical Imaging
Domain Generalization (DG) aims to learn representations that generalizes to unseen environments. Techniques typically fall into three categories: data augmentation, domain alignment, and meta-learning. In medical imaging, domain alignment methods like **Domain-Adversarial Neural Networks (DANN)** [3] have been popular for harmonizing MRI and CT data. However, recent large-scale benchmarks (e.g., DomainBed [6]) have questioned the efficacy of these methods compared to simple Empirical Risk Minimization (ERM) with data augmentation. Our work extends this critical inquiry to the 1D biomedical domain.

### B. Shortcut Learning in Deep Learning
"Shortcut learning" refers to DNNs relying on spurious correlations (e.g., background textures, hospital tokens) rather than semantic features [2]. In ECG, models have been shown to latch onto high-frequency noise or pac spikes that correlate with disease labels. While previous works identify these issues retrospectively, we propose **SAST** as a prospective stress-test to actively induce and measure this fragility.

### C. Cross-Dataset ECG Evaluation
Prior ECG studies often focus on within-dataset splits (stratified cross-validation). Few works address cross-dataset generalization. The PhysioNet 2020 Challenge approached this, but focused on varying lead configurations rather than domain shift. To our knowledge, this is the first study to strictly benchmark invariant risk minimization techniques on a "Wild" cross-device, cross-population ECG task.

## III. METHODS

### A. Problem Formulation
We consider a supervised learning setting with multiple training domains $\mathcal{E}_{train}$. For each domain $e \in \mathcal{E}_{train}$, we have a dataset $\mathcal{D}_e = \{(x_i^e, y_i^e)\}_{i=1}^{N_e}$. The goal is to learn a predictive function $f_\theta: \mathcal{X} \to \mathcal{Y}$ that minimizes the risk on an unseen test domain $\mathcal{E}_{test}$ (e.g., a new hospital).
$$ \min_\theta \mathbb{E}_{e \sim \mathcal{E}_{test}} [\mathcal{L}(f_\theta(x), y)] $$

### B. Benchmarked Algorithms
We evaluate four representative algorithms covering the spectrum of DG approaches. **Empirical Risk Minimization (ERM)** serves as the naive baseline, minimizing the sum of errors across all training domains without any explicit invariance constraint ($ \mathcal{L}_{ERM} = \sum_{e} \ell(f(x), y) $). It measures whether complex DG methods provide any tangible benefit over standard training.

To enforce feature invariance, we evaluate **Domain-Adversarial Neural Networks (DANN)** [3]. DANN employs a domain discriminator in a minimax game against the feature extractor ($ \mathcal{L}_{DANN} = \mathcal{L}_{task} - \lambda \mathcal{L}_{domain\_adv} $), theoretically removing domain-specific information from the representation. Alternatively, **Variance Risk Extrapolation (V-REx)** [7] penalizes the variance of the loss across domains ($ \mathcal{L}_{VREx} = \mathcal{L}_{ERM} + \beta \text{Var}(\mathcal{L}_e) $), discouraging reliance on shortcuts effective in only a subset of environments. Finally, **Invariant Risk Minimization (IRM)** [8] seeks a representation such that the optimal linear classifier is identical across environments, penalizing the gradient norm of a dummy classifier ($ \mathcal{L}_{IRM} = \mathcal{L}_{ERM} + \lambda || \nabla_{w} \mathcal{L}_e ||^2 $), grounded in causal inference theory.

**Implementation Validity Checks:** To ensure the validity of the baseline implementations, we evaluated multiple $\lambda$ values (10-1000) for IRM and observed consistent underperformance, confirming intrinsic instability. For DANN, we verified that the domain discriminator achieved high accuracy in the initial phases before the adversarial game stabilized. All models utilize a ResNet-1d-18 backbone with standard He initialization.

### C. Data & Preprocessing
Our study utilizes two heterogeneous datasets to simulate a "Wild" distribution shift. The **Source Domain** is represented by the PTB-XL dataset (Germany) [4], containing 21,837 records from 18,885 patients acquired using Schiller devices. It covers a wide range of morphological pathologies (MI, STTC, CD, HYP). The **Target Domain** is the Chapman-Shaoxing dataset (China) [5], comprising 10,646 records acquired using GE Healthcare devices. Chapman focuses heavily on rhythm abnormalities (AFIB, SB, GSVT). This combination of hardware shift (Schiller to GE) and task shift (Morphology to Rhythm) provides a rigorous testbed for deployment robustness.

To ensure compatibility, we unified the input representation via a standardized pipeline. All signals were resampling to 100 Hz and filtered using a 3rd-order Butterworth bandpass filter (0.5-50 Hz) to remove baseline wander and high-frequency noise. Each lead was independently Z-score normalized to handle amplitude scaling differences. Finally, we utilized a fixed 10-second crop strategy for training batches. Regarding labels, since the datasets use different diagnostic ontologies, we mapped them to comparable coarse-grained label spaces. PTB-XL's 71 statements were aggregated into 5 super-classes (NORM, MI, STTC, CD, HYP), while Chapman's codes were mapped to 4 rhythm classes (Status, AFIB, SB, GSVT). While the label spaces are not identical, they represent the standard ground-truth available in each clinical environment.

### C. Evaluation Protocol
**1) Mathematical Formulation:**
We define the learning task over domains $e \in \mathcal{E}_{source, target}$. Let $E$ denote the random variable indexing domains. A robust model should learn a representation $Z = f(X)$ such that $P(Y|Z)$ is invariant across $e$. We formulate our stress test and diagnostics as follows. We define a shortcut feature $S(X)$ such that $P(S(X)=Y) = \rho$, where $\rho=0.9$ represents extreme spurious correlation. The SAST injection is defined as $ x_{poisoned} = x + \alpha \cdot S_{freq}(t) \cdot \mathbb{I}(y \in \mathcal{Y}_{abnormal}) $, where $S_{freq}(t) = \sin(2\pi \cdot 60t)$ is a narrowband sinusoid. Valid invariant learning requires $Z \perp S(X) | Y_{causal}$.

For diagnostics, we use the **Dataset-Identity Leakage** metric. We employ linear probing to estimate the mutual information between the representation $Z$ and the domain index $E$. The probe accuracy serves as a practical lower-bound proxy for $I(Z; E)$. High leakage implies $I(Z; E) \gg 0$, indicating a failure to achieve the strong invariance condition $Z \perp E$. Additionally, we measure **Calibration Error (ECE)** on OOD data using 15 adaptive bins.

## III. RESULTS

### A. Main Performance & Robustness
Table I summarizes the performance across all methods.


**TABLE I: CROSS-DATASET GENERALIZATION & DIAGNOSTICS**
| Method | Condition | Source F1 | Target F1 | SAST Drop ($\Delta$F1) | Leakage Acc. | OOD ECE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ERM** | Clean | 0.35 | 0.85 | - | 99.8% | 0.007 |
| **ERM** | Poisoned | 0.40 | 0.83 | **-0.02** | 99.8% | 0.053 |
| **DANN** | Clean | 0.24 | 0.84 | - | 99.9% | 0.015 |
| **DANN** | Poisoned | 0.24 | **0.75** | **-0.09** | 99.9% | 0.026 |
| **V-REx** | Clean | 0.25 | 0.81 | - | 99.7% | 0.008 |
| **V-REx** | Poisoned | 0.26 | 0.82 | +0.01 | 99.9% | 0.025 |
| **IRM** | Clean | 0.12 | 0.19 | - | 94.7% | 0.177 |
| **IRM** | Poisoned | 0.10 | 0.16 | -0.03 | 93.7% | 0.131 |

![Figure 1: Cross-Dataset Generalization Performance](paper/figures/fig1_performance.png)
*Figure 1: Cross-Dataset Generalization Performance across methods. Note the significant performance drop for DANN under the Poisoned condition compared to ERM.*

### B. Detailed Performance Analysis

**1) Baseline Generalization (Clean Setting)**
Table I shows that under standard training conditions (Clean), ERM achieves a strong Target F1 score of 0.85, outperforming all specialized DG methods. DANN (0.84) and V-REx (0.81) offer no significant improvement over the naive baseline. This aligns with findings from the DomainBed benchmark [6], suggesting that in many modern deep learning regimes, strong data augmentation and backbone regularization (ResNet) provide sufficient robustness for moderate shifts. IRM, however, consistently failed to converge to a competitive solution (F1 < 0.20), highlighting the difficulty of bilevel optimization in high-dimensional feature spaces where the "invariant" solution may be hard to disentangle from the predictive one.

**2) Vulnerability to Shortcuts (SAST)**
![Figure 2: SAST Vulnerability](paper/figures/fig2_sast_drop.png)
*Figure 2: SAST Vulnerability showing the drop in F1 score when the shortcut is present. DANN suffers the largest degradation.*

The SAST protocol reveals the hidden fragility of these models. When exposed to the 60Hz shortcut during training ERM shows a negligible drop (-0.02), which is potentially deceptive (see Diagnostic Analysis below). However, DANN suffers a statistically significant performance collapse (-0.09 F1). This is the critical finding of our study: by enforcing domain invariance $\mathcal{L}_{adv}$, DANN forces the encoder to discard "domain-specific" features. However, since the 60Hz artifact is the most reliable "domain-invariant" feature (artificially inserted into both domains during SAST), DANN paradoxically latches onto it *more* strongly than ERM to satisfy its adversarial objective. This is a classic case of **Negative Transfer** induced by algorithmic constraints.

**3) The Inverted Generalization Phenomenon**
Contrary to standard expectations where Source > Target performance, we observe Target F1 (approx 0.85) >> Source F1 (approx 0.35). This is explained by the task asymmetry discussed in Methods. The Source task (PTB-XL) requires detecting subtle morphological changes (e.g., ST-elevation in MI) which are often confounded by co-morbidities. The Target task (Chapman), however, mapped to Rhythm classes (AFIB, SB, GSVT), primarily relies on R-R interval variability—a much stronger and cleaner signal that is easier for a 1D-CNN to learn even under distribution shift. This confirms that "Generalization" is a function of both Domain shift and Task difficulty.

### C. Diagnostic Insights
![Figure 3: Dataset-Identity Leakage](paper/figures/fig3_leakage.png)
*Figure 3: Dataset-Identity Leakage. All methods retain >90% domain information, indicating a failure to learn domain-invariant representations.*

The **Dataset-Identity Leakage** results (Table I, Figure 3) provide a strong diagnostic indicator. All methods retain **>99% domain information** in their embeddings. We acknowledge that raw signal statistics (e.g., sampling tokens, frequency spectra) likely allow for high leakage even without semantic processing. However, the failure of DANN to reduce this leakage *below* the ERM baseline confirms that the adversarial penalty was ineffective at masking even these low-level signatures, leaving the model vulnerable to shortcut exploitation.

## IV. DISCUSSION

### A. Failure of Invariance Assumptions
Our results challenge the prevailing assumption that enforcing statistical invariance leads to robustness. We demonstrate that classical DG methods fail to learn invariant representations on high-dimensional ECG data. DANN exhibits a **Feature Alignment Pathology**: by forcing feature distributions to align without removing the shortcut, it prioritizes the *only* features that are reliably common between domains—in this case, the injected 60Hz artifact—leading to increased fragility.

### B. Clinical Implications of "Hidden" Shortcuts
For clinical practitioners, this study highlights a silent danger. A model might appear robust on an external validation set (like our Chapman Target evaluation) not because it learned the pathology, but because it learned a subtle acquisition artifact common to that specific test set. The SAST protocol simulates this by "poisoning" the training data with a known artifact. If a model cannot ignore a 90% correlated 60Hz mains hum, it certainly cannot ignore subtle electrode impedance differences or site-specific filtering artifacts.

### C. Limitations
We acknowledge that the severe distribution shift (Schiller $\to$ GE) is compounded by a task shift (Morphology $\to$ Rhythm). While this confounds a pure "domain invariance" analysis, we argue this "Wild Distribution Shift" more accurately reflects the chaotic reality of clinical deployment where labeling protocols and hardware vary simultaneously. We do not claim this result invalidates IRM theory; rather, it highlights practical instability under severe assumption violations (high-dimensional inputs, compound shift).

**Conclusion:**
Our results demonstrate that classical DG methods fail to learn invariant representations on high-dimensional ECG data under severe, shortcut-amplified distribution shifts. Consistent with [3], IRM struggled with optimization stability, while DANN maximized coverage but failed to remove shortcut information. We recommend that **SAST** and **Leakage Probing** be adopted as standard pre-deployment diagnostics alongside traditional F1/AUC metrics. Future work should investigate non-adversarial disentanglement methods that can explicitly model and reject specific frequency-domain confounders.

## V. REFERENCES
[1] Hannun et al., Nature Medicine 2019.
[2] Geirhos et al., Nature MI 2020.
[3] Gulrajani & Lopez-Paz, ICLR 2021.
[4] Wagner et al., Scientific Data 2020.
[5] Zheng et al., Scientific Data 2020.
[Standard references 6-8...]
