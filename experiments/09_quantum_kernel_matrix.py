import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sksurv.datasets import load_gbsg2

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


# ============================================================
# 1. Quantum feature map
# ============================================================

def angle_encode(features):

    n_qubits = len(features)

    circuit = QuantumCircuit(n_qubits)

    # Feature encoding
    for i, angle in enumerate(features):
        circuit.ry(angle, i)

    # Entanglement
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
# 3. Generate all quantum states
# ============================================================

def generate_states(X):

    states = []

    for x in X:
        states.append(
            quantum_state(x)
        )

    return np.array(states)


# ============================================================
# 4. Calculate kernel matrix
# ============================================================

def quantum_kernel_matrix(states_a, states_b):

    # Matrix of inner products
    overlaps = states_a.conj() @ states_b.T

    # Fidelity-based kernel
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
# 8. Scale to [0, π]
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

print("Training states shape:")
print(train_states.shape)


print("\nGenerating testing quantum states...")

test_states = generate_states(
    X_test_scaled
)

print("Testing states shape:")
print(test_states.shape)


# ============================================================
# 10. Calculate kernel matrices
# ============================================================

print("\nCalculating training kernel matrix...")

K_train = quantum_kernel_matrix(
    train_states,
    train_states
)


print("Training kernel shape:")
print(K_train.shape)


print("\nCalculating test kernel matrix...")

K_test = quantum_kernel_matrix(
    test_states,
    train_states
)


print("Testing kernel shape:")
print(K_test.shape)


# ============================================================
# 11. Kernel diagnostics
# ============================================================

print("\n========== KERNEL DIAGNOSTICS ==========")

print("\nTraining kernel minimum:")
print(K_train.min())

print("\nTraining kernel maximum:")
print(K_train.max())

print("\nTraining kernel mean:")
print(K_train.mean())

print("\nDiagonal mean:")
print(np.mean(np.diag(K_train)))

print("\nSymmetry error:")
print(np.max(np.abs(K_train - K_train.T)))

print("\nFirst 5x5 section:")
print(K_train[:5, :5])
