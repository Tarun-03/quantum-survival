# Quantum Kernel Survival Analysis

## Project Overview

This project investigates the application of **quantum kernel methods to survival analysis**, with a particular focus on comparing quantum-kernel-based Support Vector Machines with classical survival SVM approaches.

The objective is not simply to demonstrate that a quantum model performs better than a classical model. Instead, the study aims to understand:

- How classical survival SVM methods work.
- How classical kernels such as RBF model nonlinear relationships.
- How quantum feature maps transform clinical data into quantum states.
- How different quantum encoding strategies affect the resulting kernel.
- How quantum kernels compare with classical kernels on survival prediction.
- Why a particular method performs better or worse.
- How the number of encoded features/qubits affects performance.

The current implementation uses the **GBSG2 breast cancer survival dataset** as the first experimental dataset.

---

# Research Roadmap

The planned research pipeline is:

```text
Clinical Survival Dataset
          |
          v
    Data Analysis
          |
          v
      Preprocessing
          |
          v
 Classical Survival Methods
          |
          +-------------------+
          |                   |
          v                   v
    Linear SVM          Classical RBF SVM
          |                   |
          +---------+---------+
                    |
                    v
             Baseline Results
                    |
                    v
           Quantum Preprocessing
                    |
                    v
          Quantum Feature Mapping
                    |
                    v
             Quantum Kernel
                    |
                    v
       Quantum Kernel Survival SVM
                    |
                    v
             C-index Comparison
                    |
                    v
       Encoding / Parameter Analysis
                    |
                    v
             Multiple Datasets