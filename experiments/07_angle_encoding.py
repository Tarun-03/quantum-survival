import numpy as np

from qiskit import QuantumCircuit


# ============================================================
# Angle Encoding
# ============================================================

def angle_encode(features):
    """
    Encode four classical features into four qubits
    using RY rotations followed by CNOT entanglement.
    """

    n_qubits = len(features)

    circuit = QuantumCircuit(n_qubits)

    # --------------------------------------------------------
    # Feature encoding
    # --------------------------------------------------------

    for i, angle in enumerate(features):
        circuit.ry(angle, i)

    # --------------------------------------------------------
    # Entanglement
    # --------------------------------------------------------

    for i in range(n_qubits - 1):
        circuit.cx(i, i + 1)

    return circuit


# ============================================================
# Test with one patient
# ============================================================

sample = np.array([
    1.49092533,
    0.18849556,
    0.05279988,
    0.72498292
])


circuit = angle_encode(sample)


print("========== ANGLE ENCODING ==========")

print("\nNumber of qubits:")
print(circuit.num_qubits)

print("\nInput angles:")
print(sample)

print("\nQuantum circuit:")
print(circuit)