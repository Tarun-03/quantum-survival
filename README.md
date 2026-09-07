# Quantum Kernel Survival Analysis

## Project Overview

This project investigates the application of **quantum kernel methods to survival analysis**, with a particular focus on comparing quantum-kernel-based Support Vector Machines with classical survival SVM approaches.

The main objective is to understand:

- How classical survival SVM methods work.
- How classical kernels such as RBF handle nonlinear relationships.
- Why quantum kernels may be useful for nonlinear survival analysis.
- How different quantum encoding strategies affect model performance.
- How quantum kernels compare with classical kernels.
- How the number of features and qubits affects performance.
- Why a particular model produces a high or low C-index.

The current implementation uses the **GBSG2 breast cancer survival dataset** as the first experimental dataset.

---

# Research Objective

The research follows this roadmap:

1. Understand classical SVM and regression/survival methods.
2. Establish strong classical baselines.
3. Understand why a quantum kernel may be useful.
4. Apply different quantum encoding strategies.
5. Experiment with different datasets.
6. Experiment with different SVM parameters.
7. Compare quantum and classical methods fairly.
8. Analyze the results rather than only reporting performance.
9. Explain why a particular configuration performs well or poorly.

The current work represents the **initial classical and quantum baseline stage**.

---

# Research Pipeline

```text
                    Survival Dataset
                           |
                           v
                    Dataset Analysis
                           |
                           v
                      Preprocessing
                           |
             +-------------+-------------+
             |                           |
             v                           v
      Classical Models             Quantum Processing
             |                           |
             v                           v
    Linear Survival SVM          Feature Encoding
             |                           |
             v                           v
    Classical RBF SVM             Quantum States
             |                           |
             |                           v
             |                      Quantum Kernel
             |                           |
             +-------------+-------------+
                           |
                           v
                      Survival SVM
                           |
                           v
                         C-index
                           |
                           v
                     Result Analysis
                           |
                           v
                Encoding / Parameter Analysis
                           |
                           v
                     Other Datasets
```

---

# Dataset

## GBSG2

The first dataset used in this project is the **German Breast Cancer Study Group 2 (GBSG2)** survival dataset available through `scikit-survival`.

### Dataset Size

- **686 patients**
- **8 original features**

## Original Features

```text
age
estrec
horTh
menostat
pnodes
progrec
tgrade
tsize
```

### Feature Descriptions

| Feature | Description |
|---|---|
| `age` | Patient age |
| `estrec` | Estrogen receptor level |
| `horTh` | Hormonal therapy indicator |
| `menostat` | Menopausal status |
| `pnodes` | Number of positive lymph nodes |
| `progrec` | Progesterone receptor level |
| `tgrade` | Tumor grade |
| `tsize` | Tumor size |

---

# Survival Target

The target contains two components:

```text
cens
time
```

where:

- `time` represents the observed follow-up/survival time.
- `cens` indicates whether the event was observed.

The dataset therefore contains both patients for whom the event was observed and patients whose observations were censored.

## Dataset Statistics

```text
Total patients:       686

Event patients:       299
Censored patients:    387
```

### Follow-up Statistics

```text
Minimum follow-up:     8
Maximum follow-up:     2659

Mean follow-up:        1124.49
Median follow-up:      1084
```

---

# Environment Setup

The current development environment is:

```text
Operating System: macOS
Architecture:     Apple Silicon / arm64
Python:           3.13.5

NumPy:            2.5.3
SciPy:            1.18.1
scikit-learn:     1.9.0
scikit-survival:  0.28.0
```

The project uses a Python virtual environment:

```text
.venv/
```

## Verify the Python Environment

```bash
which python
python --version
python -m pip --version
```

The Python path should point inside the project virtual environment:

```text
quantum-survival/.venv/bin/python
```

## Install scikit-survival

```bash
python -m pip install scikit-survival
```

Verify:

```bash
python -c "import sksurv; print(sksurv.__version__)"
```

Expected:

```text
0.28.0
```

## Install Qiskit

```bash
python -m pip install qiskit
```

Verify:

```bash
python -c "import qiskit; print(qiskit.__version__)"
```

## Install Qiskit Machine Learning

```bash
python -m pip install qiskit-machine-learning
```

Verify:

```bash
python -c "import qiskit; import qiskit_machine_learning; print('Qiskit:', qiskit.__version__); print('QML:', qiskit_machine_learning.__version__)"
```

## Verify the Quantum Kernel Stack

```bash
python -c "from qiskit import QuantumCircuit; from qiskit_machine_learning.kernels import FidelityQuantumKernel; print('Quantum stack OK')"
```

Current quantum experiments use Qiskit statevector simulation. No quantum hardware has been used in the completed experiments.

---

# Train/Test Split

The dataset was split into:

```text
Training samples: 548
Testing samples:  138
```

The initial experiments use:

```text
Random seed = 42
```

Later experiments use multiple random seeds to determine whether observed results are stable.

---

# Preprocessing

The original GBSG2 dataset contains categorical features.

The preprocessing pipeline converts the categorical variables into numerical representations suitable for machine learning.

The resulting processed dataset contains:

```text
12 numerical features
```

### Training Data

```text
(548, 12)
```

### Testing Data

```text
(138, 12)
```

The 12-feature representation is used for the main classical baseline.

---

# Evaluation Metric

## Concordance Index (C-index)

The primary evaluation metric is the **Concordance Index (C-index)**.

Survival analysis contains censored observations, so the problem is not treated as a normal binary classification problem.

The survival model produces a risk score/ranking, and the C-index evaluates how well the predicted ordering agrees with the observed survival outcomes.

Approximately:

```text
C-index ≈ 0.5
```

corresponds to random ranking.

Higher values indicate better concordance between predicted risk and observed survival ordering.

---

# Classical Survival SVM

Before applying quantum kernels, classical survival SVM methods were implemented.

The purpose is to establish a baseline and understand whether the quantum method provides any meaningful improvement.

The main classical models currently implemented are:

```text
1. Linear Survival SVM
2. RBF Kernel Survival SVM
```

---

# Experiment 01 — Dataset Analysis

Script:

```text
experiments/01_dataset_analysis.py
```

This experiment performs:

- Dataset loading
- Dataset shape analysis
- Feature inspection
- Data type inspection
- Missing-value analysis
- Survival-time analysis
- Event/censoring analysis

### Result

```text
Dataset:
686 patients
8 original features

Events:
299

Censored:
387
```

---

# Experiment 02 — Linear Survival SVM

Script:

```text
experiments/02_linear_survival_svm.py
```

The first classical model uses a linear kernel.

### Configuration

```text
Model:
Linear Survival SVM

Alpha:
1.0

Rank ratio:
1.0

Features:
12
```

### Result Using Random Seed 42

```text
Training samples: 548
Testing samples: 138
Features: 12

C-index:
0.6561614731
```

Therefore:

```text
Linear Survival SVM
C-index = 0.6562
```

This serves as the initial classical baseline.

---

# Experiment 03 — Classical RBF Survival SVM

Script:

```text
experiments/03_rbf_survival_svm.py
```

The second classical model uses an RBF kernel.

## Important

The RBF kernel used here is a **classical kernel**, not a quantum kernel.

The RBF kernel introduces nonlinear similarity between samples.

The main parameters investigated were:

```text
alpha
gamma
rank_ratio
```

---

# RBF Gamma Experiment

Using:

```text
alpha = 1.0
rank_ratio = 1.0
```

different gamma values were tested.

| Gamma | C-index |
|---:|---:|
| 1.0 | 0.53364 |
| 0.1 | 0.54196 |
| 0.01 | 0.66838 |
| 0.001 | **0.67440** |
| 0.0001 | 0.66041 |

The best observed value was:

```text
gamma = 0.001
```

with:

```text
C-index = 0.6743980169971672
```

---

# RBF Alpha Experiment

Using:

```text
gamma = 0.001
rank_ratio = 1.0
```

different alpha values were tested.

| Alpha | C-index |
|---:|---:|
| 0.01 | 0.66767 |
| 0.1 | 0.66625 |
| 1.0 | 0.67440 |
| 10.0 | 0.68113 |
| 100.0 | **0.68396** |

The best observed configuration was:

```text
alpha = 100
gamma = 0.001
rank_ratio = 1.0
```

with:

```text
C-index = 0.6839589235
```

This configuration is currently used as the main classical RBF baseline.

---

# Experiment 04 — Repeated Classical Experiments

Script:

```text
experiments/04_repeated_classical.py
```

To check whether the classical results depend heavily on one particular train/test split, the models were evaluated using five random seeds:

```text
42
43
44
45
46
```

## Linear Survival SVM Results

```text
Seed 42: 0.6561614731
Seed 43: 0.6733561756
Seed 44: 0.6456624193
Seed 45: 0.6299509619
Seed 46: 0.6920574163
```

Mean:

```text
0.6594376892
```

Standard deviation:

```text
0.0215765131
```

Therefore:

```text
Linear Survival SVM

Mean C-index = 0.6594
Std          = 0.0216
```

## Classical RBF Survival SVM Results

```text
Seed 42: 0.6839589235
Seed 43: 0.6970141948
Seed 44: 0.6606737651
Seed 45: 0.6318370426
Seed 46: 0.7022009569
```

Mean:

```text
0.6751369766
```

Standard deviation:

```text
0.0259729368
```

Therefore:

```text
Classical RBF Survival SVM

Mean C-index = 0.6751
Std          = 0.0260
```

---

# Classical Baseline Comparison

| Model | Mean C-index | Std |
|---|---:|---:|
| Linear Survival SVM | 0.6594 | 0.0216 |
| Classical RBF Survival SVM | **0.6751** | 0.0260 |

The classical RBF model performs better than the linear model on average.

This indicates that the nonlinear RBF representation is capturing useful structure that the linear model does not capture.

However, this does not by itself prove that the underlying clinical relationships are inherently nonlinear.

---

# Experiment 05 — Feature Analysis

Script:

```text
experiments/05_feature_analysis.py
```

Feature analysis was performed to understand which variables have stronger observed relationships with survival time and event status.

## Correlation With Survival Time

Observed correlations:

```text
pnodes    -0.256751
tsize     -0.138376
age        0.053958
estrec     0.065477
progrec    0.102729
```

The strongest observed correlation was:

```text
pnodes = -0.256751
```

followed by:

```text
tsize = -0.138376
```

## Correlation With Event

Observed correlations:

```text
progrec   -0.171596
estrec    -0.061561
age       -0.004270
tsize      0.130978
pnodes     0.242287
```

The strongest observed relationship was:

```text
pnodes = 0.242287
```

followed by:

```text
progrec = -0.171596
tsize   = 0.130978
```

These correlations are exploratory and should not be interpreted as causal relationships.

---

# Feature Statistics

| Feature | Mean | Std | Min | Median | Max |
|---|---:|---:|---:|---:|---:|
| age | 53.05 | 10.12 | 21 | 53 | 80 |
| estrec | 96.25 | 153.08 | 0 | 36 | 1144 |
| pnodes | 5.01 | 5.48 | 1 | 3 | 51 |
| progrec | 110.00 | 202.33 | 0 | 32.5 | 2380 |
| tsize | 29.33 | 14.30 | 3 | 25 | 120 |

Some features have substantially larger ranges than others, particularly:

```text
estrec
progrec
```

Therefore, scaling is required before quantum angle encoding.

---

# Experiment 06 — Quantum Preprocessing

Script:

```text
experiments/06_quantum_preprocessing.py
```

The first quantum experiment selected four features:

```text
age
pnodes
progrec
tsize
```

The reason for beginning with four features is to construct a manageable four-qubit quantum feature map and establish an initial baseline.

### Configuration

```text
Features:
4

Qubits:
4
```

The selected features were scaled to:

```text
[0, π]
```

This allows each feature to be directly used as a rotation angle.

---

# Quantum Feature Encoding

For each patient:

```text
age     → angle 1
pnodes  → angle 2
progrec → angle 3
tsize   → angle 4
```

Example encoded input:

```text
[1.49092533,
 0.18849556,
 0.05279988,
 0.72498292]
```

---

# Experiment 07 — Quantum Angle Encoding

Script:

```text
experiments/07_angle_encoding.py
```

The first quantum feature map uses:

```text
Angle Encoding
+
RY rotations
+
CNOT entanglement
```

The circuit structure is approximately:

```text
q0 ──RY(x1)──■────────────
             │
q1 ──RY(x2)──X──■─────────
                │
q2 ──RY(x3)─────X──■──────
                   │
q3 ──RY(x4)───────X
```

Implementation concept:

```python
for i, angle in enumerate(features):
    circuit.ry(angle, i)

for i in range(n_qubits - 1):
    circuit.cx(i, i + 1)
```

The circuit was verified using a statevector simulator.

---

# Quantum State Representation

For four qubits:

```text
2^4 = 16
```

Therefore each patient is represented by a statevector containing:

```text
16 amplitudes
```

Training states:

```text
(548, 16)
```

Testing states:

```text
(138, 16)
```

---

# Experiment 08 — Quantum Kernel

Script:

```text
experiments/08_quantum_kernel.py
```

The quantum kernel measures similarity between two encoded quantum states.

The kernel used is a **quantum state fidelity kernel**.

Conceptually:

```text
K(x_i, x_j) = |<ψ(x_i) | ψ(x_j)>|²
```

The value represents the similarity between the two quantum states.

For the same input:

```text
K(x, x) ≈ 1
```

---

# Example Quantum Kernel Values

The first three patients produced:

```text
K(patient 1, patient 1)
≈ 1.0000

K(patient 1, patient 2)
≈ 0.8300

K(patient 1, patient 3)
≈ 0.9736
```

Thus, according to this quantum feature map:

```text
patient 1 and patient 3
```

have greater quantum-kernel similarity than:

```text
patient 1 and patient 2
```

---

# Experiment 09 — Quantum Kernel Matrix

Script:

```text
experiments/09_quantum_kernel_matrix.py
```

The complete quantum kernel matrix was generated.

Training state matrix:

```text
(548, 16)
```

Testing state matrix:

```text
(138, 16)
```

Training kernel matrix:

```text
(548, 548)
```

Testing kernel matrix:

```text
(138, 548)
```

---

# Quantum Kernel Diagnostics — 4 Qubits

Observed values:

```text
Minimum:
9.152460931067276e-37

Maximum:
1.0000000000000013

Mean:
0.7530453009831218

Diagonal mean:
1.0

Symmetry error:
0.0
```

The symmetry error:

```text
0.0
```

confirms that the training kernel is symmetric.

The diagonal mean:

```text
1.0
```

confirms the expected self-similarity property.

---

# Experiment 10 — 4-Qubit Quantum Kernel Survival SVM

Script:

```text
experiments/10_quantum_survival_svm.py
```

The quantum kernel was supplied to the survival SVM as a precomputed kernel.

### Configuration

```text
Dataset:
GBSG2

Features:
4

Qubits:
4

Encoding:
Angle Encoding + CNOT

Kernel:
Quantum state fidelity

Alpha:
100.0

Rank ratio:
1.0
```

### Result

```text
Training samples: 548
Testing samples: 138

C-index:
0.6510269122
```

Therefore:

```text
4-Qubit Quantum Kernel Survival SVM
C-index = 0.6510
```

---

# Experiment 11 — Fair 4-Feature Classical Comparison

Script:

```text
experiments/11_four_feature_classical.py
```

The first classical RBF model used all 12 processed features, while the quantum model initially used only four features.

Therefore, a fair comparison was performed using the same four features:

```text
age
pnodes
progrec
tsize
```

### Configuration

```text
Model:
Classical RBF Survival SVM

Features:
4

Alpha:
100

Gamma:
0.001

Rank ratio:
1.0
```

### Result

```text
C-index:
0.6574008499
```

---

# Fair Classical vs Quantum Comparison

Using the same four features:

| Model | Features | C-index |
|---|---:|---:|
| Classical RBF Survival SVM | 4 | **0.6574** |
| Quantum Kernel Survival SVM | 4 | 0.6510 |

Difference:

```text
Quantum - Classical

0.6510269122 - 0.6574008499
=
-0.0063739377
```

Therefore, in this initial experiment:

```text
Classical RBF:
0.6574

Quantum:
0.6510
```

The quantum model is slightly lower than the corresponding classical RBF model.

This comparison is more meaningful than comparing the four-feature quantum model directly against the 12-feature classical model because both models use the same features.

---

# Experiment 12 — 12-Feature Quantum Kernel

Script:

```text
experiments/12_quantum_12feature.py
```

The next experiment investigated whether all 12 processed features could be directly encoded into a quantum circuit.

### Configuration

```text
Features:
12

Qubits:
12
```

The same basic encoding strategy was used:

```text
Angle Encoding + CNOT
```

---

# 12-Qubit State Dimension

For 12 qubits:

```text
2^12 = 4096
```

Therefore each patient's statevector contains:

```text
4096 amplitudes
```

Training states:

```text
(548, 4096)
```

Testing states:

```text
(138, 4096)
```

---

# 12-Qubit Kernel Matrices

Training kernel:

```text
(548, 548)
```

Testing kernel:

```text
(138, 548)
```

The kernel construction was completed successfully using statevector simulation.

---

# 12-Qubit Kernel Diagnostics

Observed:

```text
Minimum:
1.4073514691624179e-230

Maximum:
1.0000000000000013

Mean:
0.10821420365300602

Diagonal mean:
1.0

Symmetry error:
0.0
```

The most notable observation is the difference in average kernel similarity.

For four qubits:

```text
Mean kernel similarity ≈ 0.753
```

For twelve qubits:

```text
Mean kernel similarity ≈ 0.108
```

This is a major observation requiring further investigation.

---

# 12-Qubit Quantum Survival SVM

### Configuration

```text
Dataset:
GBSG2

Features:
12

Qubits:
12

Encoding:
Angle Encoding + CNOT

Alpha:
100.0

Rank ratio:
1.0

Kernel:
Quantum state fidelity
```

### Result

```text
C-index:
0.5717067989
```

Therefore:

```text
12-Feature Quantum Kernel Survival SVM
C-index = 0.5717
```

---

# Current Results Summary

| Model | Features | Qubits | Kernel | C-index |
|---|---:|---:|---|---:|
| Linear Survival SVM | 12 | — | Linear | 0.6562 |
| Classical RBF Survival SVM | 12 | — | RBF | **0.6840** |
| Classical RBF Survival SVM | 4 | — | RBF | 0.6574 |
| Quantum Kernel Survival SVM | 4 | 4 | Quantum Fidelity | 0.6510 |
| Quantum Kernel Survival SVM | 12 | 12 | Quantum Fidelity | 0.5717 |

---

# Initial Observations

## Observation 1 — RBF Improves Over the Linear Baseline

The 12-feature classical models produced:

```text
Linear:
0.6562

RBF:
0.6840
```

The repeated-seed experiment also showed:

```text
Linear:
0.6594 ± 0.0216

RBF:
0.6751 ± 0.0260
```

Therefore, the classical RBF kernel provides a stronger baseline than the linear kernel on the current dataset.

---

## Observation 2 — Feature Selection Affects Performance

The classical RBF model using 12 features achieved:

```text
0.6840
```

while the same general model using only four features achieved:

```text
0.6574
```

This indicates that removing eight features resulted in a reduction in predictive ranking performance on the current split.

Therefore, feature count must be controlled when making classical-vs-quantum comparisons.

---

## Observation 3 — 4-Qubit Quantum Performance Is Close to the Corresponding Classical Model

Using the same four features:

```text
Classical RBF:
0.6574

Quantum:
0.6510
```

Difference:

```text
≈ 0.0064
```

The difference is relatively small on this particular split.

However, repeated quantum experiments are required before determining whether this difference is statistically meaningful or simply due to the train/test split.

---

## Observation 4 — 12-Qubit Quantum Performance Drops Significantly

The 12-qubit model achieved:

```text
0.5717
```

compared with:

```text
4-qubit quantum model:
0.6510
```

At the same time, the mean quantum kernel similarity decreased from approximately:

```text
4 qubits:
0.753
```

to:

```text
12 qubits:
0.108
```

This suggests a possible relationship between the number of encoded features/qubits and the geometry of the resulting quantum kernel.

This is currently an **observation/hypothesis**, not a confirmed explanation.

---

# Working Research Hypothesis

> Increasing the number of directly encoded features and qubits using the current angle-encoding + CNOT feature map may cause pairwise quantum-state fidelities to become very small. This can change the geometry of the kernel and potentially make it less useful for survival-risk ranking.

This hypothesis needs to be tested experimentally.

The next stage therefore focuses on **kernel analysis and alternative encoding strategies**, rather than simply searching for a configuration with a higher C-index.

---

# Important Research Principle

The objective is not:

```text
Find the highest C-index
```

alone.

The objective is:

```text
Model
  ↓
Result
  ↓
Analyze result
  ↓
Understand why
  ↓
Compare against classical baseline
  ↓
Test hypothesis
```

For every experiment, the following should be recorded:

```text
Dataset
Features
Number of qubits
Encoding strategy
Kernel type
Alpha
Rank ratio
Gamma (where applicable)
Random seed
C-index
Kernel statistics
```

---

# Next Experimental Steps

## 1. Analyze Quantum Kernel Distributions

The next immediate experiment should investigate the distribution of off-diagonal quantum kernel values.

For both the four-qubit and twelve-qubit kernels, calculate:

```text
Mean
Standard deviation
Median
Minimum
Maximum
10th percentile
25th percentile
75th percentile
90th percentile
Fraction below 0.01
Fraction below 0.1
```

The goal is to determine whether the 12-qubit kernel is becoming concentrated near zero.

---

# 2. Analyze Kernel Eigenvalues

For each quantum kernel matrix, investigate:

```text
Eigenvalues
Smallest eigenvalues
Largest eigenvalues
Condition number
Effective rank
```

This can help determine whether the kernel contains useful structure or becomes poorly conditioned.

---

# 3. Repeat Quantum Experiments Across Random Seeds

The quantum models should be tested using:

```text
42
43
44
45
46
```

The final comparison should report:

```text
Mean C-index
Standard deviation
```

rather than relying on one train/test split.

---

# 4. Compare Equal Feature Sets

Two important comparisons should be performed.

## Four-Feature Comparison

```text
Classical RBF
vs
Quantum Kernel
```

using:

```text
age
pnodes
progrec
tsize
```

## Twelve-Feature Comparison

```text
Classical RBF
vs
Quantum Kernel
```

using all 12 processed features.

This will make the comparison more scientifically meaningful.

---

# 5. Test Different Quantum Encoding Strategies

The current quantum feature map is:

```text
RY Angle Encoding
+
CNOT Chain
```

Additional encoding strategies should be investigated.

One important direction is **data re-uploading**.

Instead of requiring:

```text
12 features
=
12 qubits
```

the same four qubits can potentially encode multiple groups of features.

For example:

```text
4 qubits

Features 1–4
     ↓
RY encoding
     ↓
CNOT
     ↓
Features 5–8
     ↓
RY encoding
     ↓
CNOT
     ↓
Features 9–12
```

This allows 12 features to be processed using fewer qubits.

---

# 6. Quantum Hyperparameter Analysis

The quantum survival SVM should be evaluated across:

```text
alpha
rank_ratio
```

and other relevant parameters.

The experiments should be systematic rather than based on selecting one favorable result.

---

# 7. Additional Datasets

After the GBSG2 pipeline becomes stable, additional survival datasets should be introduced.

The same experimental procedure should be followed:

```text
Dataset
   ↓
Preprocessing
   ↓
Classical Linear Baseline
   ↓
Classical RBF Baseline
   ↓
Quantum Encoding
   ↓
Quantum Kernel
   ↓
Quantum Survival SVM
   ↓
C-index
   ↓
Kernel Analysis
```

This will help determine whether the observations from GBSG2 generalize to other datasets.

---

# 8. Final Comparison

The final study should compare:

```text
Classical Linear Kernel
        vs
Classical RBF Kernel
        vs
Quantum Kernel A
        vs
Quantum Kernel B
        vs
Quantum Kernel C
```

across:

```text
Multiple datasets
Multiple random seeds
Multiple feature configurations
Multiple quantum encoding strategies
Multiple SVM parameters
```

The final analysis should explain not only **which model performs best**, but also **why the performance differs**.

---

# Completed Experiments

```text
[x] GBSG2 dataset loading
[x] Dataset inspection
[x] Missing-value analysis
[x] Survival/event analysis
[x] Train/test preprocessing
[x] Classical Linear Survival SVM
[x] Classical RBF Survival SVM
[x] RBF gamma experiments
[x] RBF alpha experiments
[x] Multiple random seed classical experiments
[x] Feature correlation analysis
[x] Quantum preprocessing
[x] Quantum angle encoding
[x] Quantum statevector simulation
[x] Quantum kernel calculation
[x] Quantum kernel matrix construction
[x] Quantum kernel diagnostics
[x] 4-qubit Quantum Kernel Survival SVM
[x] Fair 4-feature Classical RBF comparison
[x] 12-qubit Quantum Kernel Survival SVM
```

---

# Current Repository Structure

```text
quantum-survival/
│
├── README.md
│
├── src/
│   └── preprocessing.py
│
├── experiments/
│   ├── 01_dataset_analysis.py
│   ├── 02_linear_survival_svm.py
│   ├── 03_rbf_survival_svm.py
│   ├── 04_repeated_classical.py
│   ├── 05_feature_analysis.py
│   ├── 06_quantum_preprocessing.py
│   ├── 07_angle_encoding.py
│   ├── 08_quantum_kernel.py
│   ├── 09_quantum_kernel_matrix.py
│   ├── 10_quantum_survival_svm.py
│   ├── 11_four_feature_classical.py
│   └── 12_quantum_12feature.py
│
└── results/
```

---

# Technology Stack

The current implementation uses:

```text
Python
NumPy
SciPy
scikit-learn
scikit-survival
Qiskit
Qiskit Machine Learning
```

Quantum experiments currently use:

```text
Qiskit Statevector simulation
```

No quantum hardware has been used in the experiments completed so far.

---

# Reproducibility

Experiments are maintained as separate scripts under:

```text
experiments/
```

Rather than overwriting previous experiments, each experiment is retained as a separate stage of the research pipeline.

This makes it possible to reproduce the progression from:

```text
Classical baseline
        ↓
Classical nonlinear kernel
        ↓
Quantum preprocessing
        ↓
Quantum kernel
        ↓
Quantum survival SVM
```

---

# Current Conclusion

The initial experiments successfully establish both classical and quantum survival-analysis pipelines on the GBSG2 dataset.

The strongest classical result currently observed is:

```text
Classical RBF Survival SVM

Seed-42 C-index:
0.6840

Five-seed mean:
0.6751 ± 0.0260
```

The initial four-qubit quantum model achieved:

```text
Quantum Kernel Survival SVM

C-index:
0.6510
```

while the corresponding four-feature classical RBF model achieved:

```text
C-index:
0.6574
```

The twelve-qubit quantum model achieved:

```text
C-index:
0.5717
```

The twelve-qubit quantum kernel also showed a much lower mean pairwise similarity:

```text
4 qubits:
≈ 0.753

12 qubits:
≈ 0.108
```

These results do not yet establish a quantum advantage.

Instead, they provide the baseline observations required for the next stage of the research.

The next objective is to investigate:

```text
Why does the quantum kernel behave differently
as the number of encoded features increases?

How does the quantum encoding strategy affect
kernel geometry?

Can alternative quantum feature maps improve
the survival ranking?

Can fewer qubits encode more features effectively?

How do quantum kernels compare with classical
kernels under equal experimental conditions?
```

The research will therefore proceed with **kernel-geometry analysis, repeated quantum experiments, alternative encoding strategies, hyperparameter analysis, and additional survival datasets**.
