import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sksurv.datasets import load_gbsg2


# ============================================================
# 1. Load dataset
# ============================================================

X, y = load_gbsg2()


# ============================================================
# 2. Select 4 features for first quantum experiment
# ============================================================

quantum_features = [
    "age",
    "pnodes",
    "progrec",
    "tsize"
]


# ============================================================
# 3. Train/Test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# 4. Extract selected features
# ============================================================

X_train_quantum = X_train[quantum_features].copy()
X_test_quantum = X_test[quantum_features].copy()


# ============================================================
# 5. Scale features to [0, π]
# ============================================================

scaler = MinMaxScaler(
    feature_range=(0, np.pi)
)

X_train_angles = scaler.fit_transform(
    X_train_quantum
)

X_test_angles = scaler.transform(
    X_test_quantum
)


# ============================================================
# 6. Print information
# ============================================================

print("========== QUANTUM INPUT PREPARATION ==========")

print("\nSelected features:")
print(quantum_features)

print("\nNumber of qubits:")
print(len(quantum_features))

print("\nTraining shape:")
print(X_train_angles.shape)

print("\nTesting shape:")
print(X_test_angles.shape)

print("\nTraining angle ranges:")

for i, feature in enumerate(quantum_features):

    print(
        f"{feature}: "
        f"{X_train_angles[:, i].min():.4f} "
        f"to "
        f"{X_train_angles[:, i].max():.4f}"
    )

print("\nFirst 5 encoded training samples:")
print(X_train_angles[:5])