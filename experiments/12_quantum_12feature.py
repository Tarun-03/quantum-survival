import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sksurv.datasets import load_gbsg2
from sksurv.svm import FastKernelSurvivalSVM
from sksurv.metrics import concordance_index_censored

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


# ============================================================
# 1. Quantum feature map
# ============================================================

def angle_encode(features):

    n_qubits = len(features)

    circuit = QuantumCircuit(n_qubits)

    # Encode one feature into each qubit
    for i, angle in enumerate(features):
        circuit.ry(angle, i)

    # Linear CNOT entanglement
    for i in range(n_qubits - 1):
        circuit.cx(i, i + 1)

    return circuit


# ============================================================
# 2. Convert one patient into a quantum state
# ============================================================

def quantum_state(features):

    circuit = angle_encode(features)

    state = Statevector.from_instruction(circuit)

    return state.data


# ============================================================
# 3. Generate quantum states
# ============================================================

def generate_states(X):

    states = []

    for i, x in enumerate(X):

        if i % 100 == 0:
            print(f"Generating state {i}/{len(X)}")

        states.append(
            quantum_state(x)
        )

    return np.array(states)


# ============================================================
# 4. Quantum kernel
# ============================================================

def quantum_kernel_matrix(states_a, states_b):

    # <psi_a | psi_b>
    overlaps = states_a.conj() @ states_b.T

    # Fidelity kernel
    kernel = np.abs(overlaps) ** 2

    return kernel.real


# ============================================================
# 5. Load GBSG2
# ============================================================

print("Loading GBSG2 dataset...")

X, y = load_gbsg2()


# ============================================================
# 6. Train/test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# 7. Convert categorical variables to numerical
# ============================================================

print("\nPreprocessing features...")


# Create copies so original data is not modified
X_train = X_train.copy()
X_test = X_test.copy()


# ------------------------------------------------------------
# horTh
# no  -> 0
# yes -> 1
# ------------------------------------------------------------

X_train["horTh"] = (
    X_train["horTh"]
    .map({"no": 0, "yes": 1})
)

X_test["horTh"] = (
    X_test["horTh"]
    .map({"no": 0, "yes": 1})
)


# ------------------------------------------------------------
# menostat
# Pre -> 0
# Post -> 1
# ------------------------------------------------------------

X_train["menostat"] = (
    X_train["menostat"]
    .map({"Pre": 0, "Post": 1})
)

X_test["menostat"] = (
    X_test["menostat"]
    .map({"Pre": 0, "Post": 1})
)


# ------------------------------------------------------------
# tgrade
#
# I   -> 1
# II  -> 2
# III -> 3
# ------------------------------------------------------------

X_train["tgrade"] = (
    X_train["tgrade"]
    .map({"I": 1, "II": 2, "III": 3})
)

X_test["tgrade"] = (
    X_test["tgrade"]
    .map({"I": 1, "II": 2, "III": 3})
)


# ============================================================
# 8. Convert to numerical arrays
# ============================================================

feature_names = [
    "age",
    "estrec",
    "horTh",
    "menostat",
    "pnodes",
    "progrec",
    "tgrade",
    "tsize"
]

X_train_numeric = X_train[feature_names].astype(float)
X_test_numeric = X_test[feature_names].astype(float)


# ============================================================
# 9. Add one-hot encoded categorical features
# ============================================================
#
# To obtain the same 12-dimensional representation:
#
# horTh     -> 2 columns
# menostat  -> 2 columns
# tgrade    -> 3 columns
#
# But we already have the numeric versions above.
#
# Instead, use one-hot encoding explicitly.
# ============================================================

from sklearn.preprocessing import OneHotEncoder

categorical_features = [
    "horTh",
    "menostat",
    "tgrade"
]

numerical_features = [
    "age",
    "estrec",
    "pnodes",
    "progrec",
    "tsize"
]


encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)

X_train_cat = encoder.fit_transform(
    X_train[categorical_features]
)

X_test_cat = encoder.transform(
    X_test[categorical_features]
)


# Numerical features
X_train_num = X_train[numerical_features].astype(float)
X_test_num = X_test[numerical_features].astype(float)


# Combine numerical + categorical
X_train_processed = np.hstack([
    X_train_num.values,
    X_train_cat
])

X_test_processed = np.hstack([
    X_test_num.values,
    X_test_cat
])


# ============================================================
# 10. Check feature dimensions
# ============================================================

print("\n========== PREPROCESSING ==========")

print("Training shape:")
print(X_train_processed.shape)

print("\nTesting shape:")
print(X_test_processed.shape)

print("\nNumber of features:")
print(X_train_processed.shape[1])


# ============================================================
# 11. Scale all 12 features to [0, π]
# ============================================================

scaler = MinMaxScaler(
    feature_range=(0, np.pi)
)

X_train_scaled = scaler.fit_transform(
    X_train_processed
)

X_test_scaled = scaler.transform(
    X_test_processed
)


print("\n========== QUANTUM ENCODING ==========")

print("Number of qubits:")
print(X_train_scaled.shape[1])

print("\nAngle range:")

print(
    "Minimum:",
    X_train_scaled.min()
)

print(
    "Maximum:",
    X_train_scaled.max()
)


# ============================================================
# 12. Generate quantum states
# ============================================================

print("\nGenerating training quantum states...")

train_states = generate_states(
    X_train_scaled
)

print("\nTraining states shape:")
print(train_states.shape)


print("\nGenerating testing quantum states...")

test_states = generate_states(
    X_test_scaled
)

print("\nTesting states shape:")
print(test_states.shape)


# ============================================================
# 13. Build quantum kernel matrices
# ============================================================

print("\nBuilding quantum training kernel...")

K_train = quantum_kernel_matrix(
    train_states,
    train_states
)

print("Training kernel shape:")
print(K_train.shape)


print("\nBuilding quantum test kernel...")

K_test = quantum_kernel_matrix(
    test_states,
    train_states
)

print("Testing kernel shape:")
print(K_test.shape)


# ============================================================
# 14. Kernel diagnostics
# ============================================================

print("\n========== KERNEL DIAGNOSTICS ==========")

print("Minimum:")
print(K_train.min())

print("\nMaximum:")
print(K_train.max())

print("\nMean:")
print(K_train.mean())

print("\nDiagonal mean:")
print(np.mean(np.diag(K_train)))

print("\nSymmetry error:")
print(
    np.max(
        np.abs(K_train - K_train.T)
    )
)


# ============================================================
# 15. Train Quantum Kernel Survival SVM
# ============================================================

print("\nTraining Quantum Kernel Survival SVM...")

model = FastKernelSurvivalSVM(
    alpha=100.0,
    rank_ratio=1.0,
    kernel="precomputed",
    max_iter=10000
)

model.fit(
    K_train,
    y_train
)


# ============================================================
# 16. Predict risk scores
# ============================================================

risk_scores = model.predict(
    K_test
)


# ============================================================
# 17. Calculate C-index
# ============================================================

cindex = concordance_index_censored(
    y_test["cens"],
    y_test["time"],
    risk_scores
)[0]


# ============================================================
# 18. Final results
# ============================================================

print("\n========================================")
print("FINAL RESULTS")
print("========================================")

print("Dataset: GBSG2")

print("Model: Quantum Kernel Survival SVM")

print("Encoding: Angle Encoding + CNOT")

print("Number of qubits:", X_train_scaled.shape[1])

print("Number of features:", X_train_scaled.shape[1])

print("Alpha:", 100.0)

print("Rank ratio:", 1.0)

print("Training samples:", len(X_train_scaled))

print("Testing samples:", len(X_test_scaled))

print("\nQuantum C-index:")
print(cindex)

print("========================================")