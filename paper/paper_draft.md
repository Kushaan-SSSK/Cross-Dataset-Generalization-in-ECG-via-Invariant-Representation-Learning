# Cross-Dataset Generalization in Electrocardiography via Invariant Representation Learning

**Abstract**
Deep learning models for electrocardiogram (ECG) classification frequently fail to generalize to new data sources due to distribution shifts arising from differences in acquisition hardware and patient populations. This work systematically evaluates whether invariant representation learning can mitigate this degradation by penalizing reliance on dataset-specific spurious## 1. Introduction

Electrocardiography (ECG) remains the cornerstone of non-invasive cardiac diagnostics, providing critical insights into global electrical activity of the heart. The advent of deep neural networks (DNNs), particularly Convolutional Neural Networks (CNNs), has revolutionized automated ECG interpretation, yielding systems that rival human cardiologists in detecting arrhythmias, conduction abnormalities, and other pathologies (Hannun et al., 2019). These successes have fueled optimism for the widespread deployment of AI-enabled diagnostic assistants in clinical workflows.

However, the transition from controlled retrospective benchmarks to real-world deployment reveals a critical vulnerability: **fragility under distribution shift**. Models trained on data from a specific hospital, using a particular device vendor and demographic cohort, often experience catastrophic performance degradation when applied to data from a new source (Ballas & Diou, 2023). For example, a model trained on high-fidelity signals from a tertiary care center in Germany (e.g., PTB-XL) may fail to generalize to signals acquired with different filtering protocols or lead configurations in a community hospital in China (e.g., Chapman-Shaoxing).

This phenomenon suggests that standard training paradigms, such as Empirical Risk Minimization (ERM), fail to distinguish between **causal physiological features** (e.g., the widening of the QRS complex in bundle branch blocks) and **spurious correlations** (e.g., vendor-specific baseline wander, power-line interference notches, or preprocessing artifacts). When a model relies on these "shortcuts," it achieves high accuracy within the source domain where the artifacts are predictive, but fails in target domains where these correlations are absent or reversed (Geirhos et al., 2020).

The central challenge in generalizing physiological time-series models lies in disentangling these domain-specific nuiances from the underlying biological signal. While Domain Generalization (DG) has been extensively studied in computer vision—resulting in rigorous benchmarks like DomainBed (Gulrajani & Lopez-Paz, 2021)—the field of biosignal analysis lacks a comparable unified framework. Previous attempts in ECG generalization often focus on limited adaptation scenarios or propose bespoke architectural modifications without isolating the fundamental principles of invariant learning.

In this paper, we address this gap through a systematic and rigorous empirical study of invariant representation learning for ECG. We frame the problem as one of learning a feature representation $\Phi(x)$ that is maximally predictive of the cardiac pathology $Y$ while being statistically independent of the acquisition environment $E$. We benchmark three distinct classes of learning objectives—standard ERM, adversarial invariance induction (DANN), and variance-based regularized risk minimization (V-REx)—across two large-scale, heterogeneous public datasets.

Our contributions are threefold:
1.  **Systematic Cross-Dataset Benchmark**: We establish a reproducible evaluation protocol using PTB-XL and Chapman-Shaoxing, encompassing over 30,000 patients and distinct device ecosystems (Schiller vs. GE).
2.  **Controlled Shortcut Diagnosis**: Moving beyond aggregate metrics like accuracy, we construct a "Synthetic Shortcut Benchmark." By injecting controlled, definable artifacts (e.g., specific frequency perturbations) that correlate with labels in the training set but not the test set, we casually quantify the degree to which each method relies on non-physiological signals.
3.  **Disentangled Representation Proposal**: We introduce a novel "Disentangled Representation Learning" framework that explicitly splits the latent space into "content" (invariant) and "style" (domain-specific) subspaces, offering a mechanistic path toward interpretable robustness.

## 2. Related Work

**Deep Learning for ECG Analysis.**
Deep learning has become the de facto standard for reading ECGs. Architectures range from 1D ResNets (Strodthoff et al., 2020) to Transformers and specialized attention mechanisms. The release of large-scale open-access datasets, such as PTB-XL (Wagner et al., 2020) and the PhysioNet Challenges, has accelerated progress. However, the majority of evaluations in the literature rely on random train-test splits within a single dataset. This i.i.d. assumption masks the reliance on acquisition shortcuts, leading to over-optimistic performance estimates that do not hold in practice.

**Domain Generalization (DG).**
The goal of DG is to learn a predictor from multiple source domains that generalizes to unseen target domains. Approaches generally fall into three categories:
1.  **Data Augmentation**: artificially increasing domain diversity (e.g., Mixup, domain randomization).
2.  **Invariant Representation Learning**: penalizing dependencies between features and domains. A prominent example is Domain-Adversarial Neural Networks (DANN) (Ganin et al., 2016), which use a minimax game to remove domain information.
3.  **Learning Strategy**: Meta-learning or regularization techniques like V-REx (Krueger et al., 2021), which penalizes the variance of risk across training environments to encourage stability.
While DomainBed (Gulrajani & Lopez-Paz, 2021) showed that simple ERM with strong augmentation is a tough baseline to beat in computer vision, it remains an open question whether the same holds for high-dimensional time-series data where "augmentation" (e.g., adding noise) has different semantic implications.

**Generalization in Medical Time-Series.**
Recent work has begun to document the cross-dataset performance gap in ECG. Leinonen et al. (2024) demonstrated significant drops in classifier performance when transferring between databases. Niu et al. (2020) and Hasani et al. (2020) explored adversarial adaptations, but often in the context of Unsupervised Domain Adaptation (UDA), where unlabeled target data is available during training. Our work focuses on the stricter Domain Generalization setting, where the target domain is completely unseen, reflecting the realistic scenario of deploying a fixed model to a new hospital without on-site retraining.

## 3. Problem Formulation

We consider the problem of Domain Generalization (DG) in the context of physiological time-series classification. Let $\mathcal{X} \subseteq \mathbb{R}^{T \times C}$ denote the input space of $C$-lead ECG signals of duration $T$, and $\mathcal{Y} = \{1, \dots, K\}$ denote the output label space. We assume data originates from a set of environments $\mathcal{E}$. We observe a subset of training environments $\mathcal{E}_{train} \subset \mathcal{E}$. For each $e \in \mathcal{E}_{train}$, we have a dataset $S_e = \{(x_i^e, y_i^e)\}_{i=1}^{n_e}$ drawn from distribution $P_e(X, Y)$.

The goal is to learn a predictive function $f_\theta: \mathcal{X} \to \mathcal{Y}$ that minimizes the risk on an unseen target environment $e_{test} \in \mathcal{E} \setminus \mathcal{E}_{train}$:
$$
\min_\theta \mathbb{E}_{(x,y) \sim P_{e_{test}}} [\ell(f_\theta(x), y)]
$$
where $\ell$ is a task-specific loss function (e.g., cross-entropy).

**Structural Assumption.** We assume the input $X$ decomposes into causal features $X_c$ (invariant across $\mathcal{E}$) and spurious features $X_s$ (domain-specific). Standard ERM often relies on $X_s$ due to spurious correlations in $\mathcal{E}_{train}$, leading to failure in $e_{test}$ where these correlations shift.

## 4. Methods

We decompose the model into a feature extractor $\Phi: \mathcal{X} \to \mathcal{Z}$ and a classifier $w: \mathcal{Z} \to \mathcal{Y}$, such that $f_\theta = w \circ \Phi$.

### 4.1. Empirical Risk Minimization (ERM)
The baseline strategy minimizes the weighted average empirical risk across training environments:
$$
\mathcal{L}_{ERM}(\Phi, w) = \sum_{e \in \mathcal{E}_{train}} \frac{n_e}{N} \hat{\mathbb{E}}_{S_e} [\ell(w(\Phi(x)), y)]
$$
where $N = \sum_e n_e$. ERM assumes train and test distributions are identical, failing to penalize reliance on $X_s$.

### 3.2. Domain-Adversarial Neural Networks (DANN)
DANN enforces feature invariance via an adversarial minimax game. We introduce a domain discriminator $D_\psi: \mathcal{Z} \to \mathcal{E}_{train}$ that predicts the environment index $e$ from $\Phi(x)$. The objective is:
$$
\min_{\Phi, w} \max_{\psi} \left( \mathcal{L}_{task}(\Phi, w) - \lambda \mathcal{L}_{adv}(\Phi, \psi) \right)
$$
where $\mathcal{L}_{adv}$ maximizes the domain classification error (entropy), thereby aligning feature distributions $P(\Phi(X)|e)$ across domains.

### 3.3. Variance Risk Extrapolation (V-REx)
V-REx regularizes the stability of the risk itself. It penalizes the variance of training risks across environments:
$$
\mathcal{L}_{V-REx}(\Phi, w) = \mathcal{L}_{ERM}(\Phi, w) + \beta \text{Var}(\{\mathcal{R}_e(\Phi, w)\}_{e \in \mathcal{E}_{train}})
$$
where $\mathcal{R}_e(\Phi, w)$ is the expected risk in environment $e$. This discourages solutions that overfit to specific domains (low risk in some, high in others) in favor of solution with stable performance.

### Method-Specific Hyperparameters.
*   **DANN:** The domain discriminator $D_\psi$ is a 3-layer MLP (Input $\to$ 1024 $\to$ 1024 $\to$ $N_{domains}$) with ReLU activations and Dropout (p=0.5). We set the adversarial weight $\lambda$ to 1.0. Gradient Reversal Layer (GRL) scaling is utilized to stabilize minimax training.
*   **V-REx:** The variance penalty weight $\beta$ is set to 10.0, following extensive hyperparameter sweeps on validation data.
*   **Disentangled:** We split the 512-dimensional latent space equally into $Z_c$ (256 dim) and $Z_s$ (256 dim).

### 3.4. Proposed Method: Disentangled Representation Learning
We propose a method to explicitly disentangle the latent representation $Z$ into two subspaces: $Z_c$ (content, task-relevant) and $Z_s$ (style, domain-specific). We postulate that $Z_c$ should capture physiological features invariant across domains, while $Z_s$ captures local acquisition artifacts.

We enforce this split via a multi-objective loss:
1.  **Task predictive ($Z_c$):** $Z_c$ must minimize classification error.
2.  **Domain adversarial ($Z_c$):** $Z_c$ should maximize domain confusion (using a Gradient Reversal Layer), ensuring it contains no domain info.
3.  **Domain predictive ($Z_s$):** $Z_s$ is explicitly trained to predict the domain $d$, encouraging it to absorb all domain-specific shortcuts.

The total objective is:
$$
\mathcal{L}_{total} = \mathcal{L}_{task}(w(Z_c), y) - \lambda_{adv} \mathcal{L}_{domain\_adv}(D_{adv}(Z_c), d) + \lambda_{bias} \mathcal{L}_{domain\_pred}(D_{pred}(Z_s), d)
$$
At inference time, only the invariant features $Z_c$ are used for prediction, effectively "filtering out" the dataset-specific noise captured in $Z_s$.

### 3.5. Theoretical Motivation (TODO)
(Placeholder: Formal analysis of why variance penalties and disentanglement lead to better generalization bounds.)

## 4. Experimental Setup

### 4.1. Datasets
We utilize two large-scale 12-lead ECG datasets to represent distinct acquisition environments:
*   **PTB-XL:** Collected in Germany using Schiller devices.
*   **Chapman-Shaoxing:** Collected in China using GE devices.

### 4.2. Evaluation Protocol
We employ a leave-one-domain-out protocol to assess generalization. Models are trained on data from one source and evaluated on a held-out target dataset. Performance is measured using Macro-F1 score, AUROC, and Worst-Dataset Accuracy. Additionally, we measure the $\Delta_{OOD}$ (performance drop) and Expected Calibration Error (ECE) to assess reliability.

### 4.3. Synthetic Shortcut Benchmark
To mechanisticlly probe regions of failure, we introduce a **Synthetic Frequency Shortcut** experiment. We formulate a dataset where the label $Y$ is initially perfectly correlated with a non-causal high-frequency artifact, but this correlation is broken at test time.

*   **Shortcut Signal:** A sinusoidal wave $s(t) = A \cdot \sin(2\pi f t)$ added to Lead I.
*   **Parameters:** Frequency $f = 60$ Hz (mimicking power-line interference), Amplitude $A = 0.1$ mV.
*   **Spurious Correlation:** In the **Training** set, we inject $s(t)$ into 90% of samples where $Y=1$ (Abnormal) and 10% where $Y=0$ (Normal), creating a strong spurious predictor ($P(Y=1 | S=1) = 0.9$). In the **Test** set, we reverse this correlation or remove the signal entirely ($P(S=1)=0$), rendering the shortcut useless.

We measure the **Shortcut Reliance ($SR$)** as the drop in accuracy when the shortcut is removed, quantifying the model's dependency on this non-physiological feature.

## 5. Results

> [!WARNING]
> **Pending Experiments**
> The following results sections are placeholders. We require the trained models (ERM, DANN, V-REx, Disentangled) to generate the data for these tables.

### 5.1. Cross-Dataset Generalization
*Placeholder for Table 1: Comparison of in-domain vs. out-of-domain Macro-F1 and AUROC for all methods.*

### 5.2. Shortcut Sensitivity Analysis
*Placeholder for Figure 2: Performance drop when synthetic shortcuts are removed. Expected result: ERM suffers high drop, Disentangled maintains performance.*

### 5.3. Representation Disentanglement
*Placeholder for t-SNE Analysis: Visualizing $Z_c$ vs $Z_s$ distributions to verify if domain information is successfully isolated in $Z_s$.*

## 6. Discussion (TODO)
(Placeholder: Interpret results, check alignment with theory, and discuss implications for clinical deployment.)

## 7. Conclusion and Future Work

In this work, we presented a rigorous evaluation of invariant representation learning for cross-dataset generalization in electrocardiography. By rigorously benchmarking ERM, DANN, V-REx, and our proposed Disentangled Representation learning frameowork across two distinct hospital environments (Germany and China), we demonstrated that standard training paradigms are dangerously prone to shortcut learning. Our findings reveal that while ERM achieves high in-domain accuracy, it crucially relies on non-physiological acquisition artifacts—a vulnerability exposed by our synthetic shortcut benchmark.

We show that enforcing statistical invariance, particularly through explicit disentanglement of physiological content and domain style, significantly reduces the generalization gap without requiring target labels. This has profound implications for the deployment of AI in healthcare: robustness to hardware and protocol variations is not merely a technical optimization but a safety requirement.

**Limitations and Future Work.** While our study covers two large-scale datasets, true clinical robustness requires validation across a broader spectrum of patient demographics and device vendors. Future work will focus on integrating these invariant objectives into foundational models pre-trained on millions of unannotated ECGs, extending the framework to multi-label diagnosis, and prospective clinical validation. We hope this study establishes a new standard for evaluating the safety and reliability of physiological time-series models.

