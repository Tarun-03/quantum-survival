import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sksurv.datasets import load_gbsg2
from sksurv.svm import FastKernelSurvivalSVM

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


# ============================================================
# 1. Quantum feature map
# ============================================================

def angle_encode(features):

    n_qubits = len(features)

    circuit = QuantumCircuit(n_qubits)

    # Angle encoding
    for i, angle in enumerate(features):
        circuit.ry(angle, i)

    # CNOT entanglement
    for i in range(n_qubits - 1):
        circuit.cx(i, i + 1)

    return circuit


# ============================================================
# 2. Generate quantum state
# ============================================================

def quantum_state(features):

    circuit = angle_encode(features)

    return Statevector.from_instruction(circuit).data


# ============================================================
# 3. Generate states for all samples
# ============================================================

def generate_states(X):

    states = []

    for x in X:
        states.append(
            quantum_state(x)
        )

    return np.array(states)


# ============================================================
# 4. Quantum kernel matrix
# ============================================================

def quantum_kernel_matrix(states_a, states_b):

    overlaps = states_a.conj() @ states_b.T

    kernel = np.abs(overlaps) ** 2

    return kernel.real


# ============================================================
# 5. Load GBSG2
# ============================================================

X, y = load_gbsg2()


# ============================================================
# 6. Select quantum features
# ============================================================

quantum_features = [
    "age",
    "pnodes",
    "progrec",
    "tsize"
]


# ============================================================
# 7. Train/test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# 8. Scale features
# ============================================================

scaler = MinMaxScaler(
    feature_range=(0, np.pi)
)

X_train_scaled = scaler.fit_transform(
    X_train[quantum_features]
)

X_test_scaled = scaler.transform(
    X_test[quantum_features]
)


# ============================================================
# 9. Generate quantum states
# ============================================================

print("Generating training quantum states...")

train_states = generate_states(
    X_train_scaled
)

print("Generating testing quantum states...")

test_states = generate_states(
    X_test_scaled
)


# ============================================================
# 10. Build kernel matrices
# ============================================================

print("Building quantum training kernel...")

K_train = quantum_kernel_matrix(
    train_states,
    train_states
)

print("Building quantum test kernel...")

K_test = quantum_kernel_matrix(
    test_states,
    train_states
)


# ============================================================
# 11. Train Quantum Kernel Survival SVM
# ============================================================

print("\nTraining Quantum Kernel Survival SVM...")

model = FastKernelSurvivalSVM(
    alpha=100.0,
    rank_ratio=1.0,
    kernel="precomputed"
)

model.fit(
    K_train,
    y_train
)


# ============================================================
# 12. Predict risk scores
# ============================================================

risk_scores = model.predict(
    K_test
)


# ============================================================
# 13. Calculate C-index
# ============================================================

from sksurv.metrics import concordance_index_censored

cindex = concordance_index_censored(
    y_test["cens"],
    y_test["time"],
    risk_scores
)[0]


# ============================================================
# 14. Results
# ============================================================

print("\n========== RESULTS ==========")

print("Dataset: GBSG2")

print("Model: Quantum Kernel Survival SVM")

print("Encoding: Angle Encoding + CNOT")

print("Qubits:", len(quantum_features))

print("Alpha:", 100.0)

print("Rank ratio:", 1.0)

print("Kernel: Quantum state fidelity")

print("\nTraining samples:", len(X_train))

print("Testing samples:", len(X_test))

print("Features:", len(quantum_features))

print("\nC-index:", cindex)