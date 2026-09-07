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

    # Encode features using RY rotations
    for i, angle in enumerate(features):
        circuit.ry(angle, i)

    # CNOT entanglement
    for i in range(n_qubits - 1):
        circuit.cx(i, i + 1)

    return circuit


# ============================================================
# 2. Convert classical features → quantum state
# ============================================================

def quantum_state(features):

    circuit = angle_encode(features)

    return Statevector.from_instruction(circuit)


# ============================================================
# 3. Quantum kernel
# ============================================================

def quantum_kernel(x1, x2):

    state1 = quantum_state(x1)
    state2 = quantum_state(x2)

    overlap = np.vdot(
        state1.data,
        state2.data
    )

    return float(np.abs(overlap) ** 2)


# ============================================================
# 4. Load GBSG2
# ============================================================

X, y = load_gbsg2()


# ============================================================
# 5. Select four quantum features
# ============================================================

quantum_features = [
    "age",
    "pnodes",
    "progrec",
    "tsize"
]


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
# 7. Scale features to [0, π]
# ============================================================

scaler = MinMaxScaler(
    feature_range=(0, np.pi)
)

X_train_quantum = scaler.fit_transform(
    X_train[quantum_features]
)


# ============================================================
# 8. Select three patients
# ============================================================

patient_1 = X_train_quantum[0]
patient_2 = X_train_quantum[1]
patient_3 = X_train_quantum[2]


# ============================================================
# 9. Calculate quantum similarities
# ============================================================

k_11 = quantum_kernel(
    patient_1,
    patient_1
)

k_12 = quantum_kernel(
    patient_1,
    patient_2
)

k_13 = quantum_kernel(
    patient_1,
    patient_3
)


# ============================================================
# 10. Print results
# ============================================================

print("========== QUANTUM KERNEL ==========")

print("\nNumber of qubits:")
print(4)

print("\nFeatures:")
print(quantum_features)

print("\nPatient 1 angles:")
print(patient_1)

print("\nPatient 2 angles:")
print(patient_2)

print("\nPatient 3 angles:")
print(patient_3)

print("\nKernel similarities:")

print("K(patient 1, patient 1):", k_11)
print("K(patient 1, patient 2):", k_12)
print("K(patient 1, patient 3):", k_13)