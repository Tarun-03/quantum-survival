import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sksurv.datasets import load_gbsg2
from sksurv.svm import FastKernelSurvivalSVM
from sksurv.metrics import concordance_index_censored


# ============================================================
# 1. Load dataset
# ============================================================

X, y = load_gbsg2()


# ============================================================
# 2. Same four features used by quantum experiment
# ============================================================

features = [
    "age",
    "pnodes",
    "progrec",
    "tsize"
]


# ============================================================
# 3. Train/test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# 4. Extract features
# ============================================================

X_train = X_train[features].astype(float)
X_test = X_test[features].astype(float)


# ============================================================
# 5. Standardize
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 6. Train classical RBF Survival SVM
# ============================================================

print("Training 4-feature Classical RBF Survival SVM...")

model = FastKernelSurvivalSVM(
    alpha=100.0,
    gamma=0.001,
    rank_ratio=1.0,
    kernel="rbf",
    max_iter=10000

)

model.fit(
    X_train_scaled,
    y_train
)


# ============================================================
# 7. Predict
# ============================================================

risk_scores = model.predict(
    X_test_scaled
)


# ============================================================
# 8. C-index
# ============================================================

cindex = concordance_index_censored(
    y_test["cens"],
    y_test["time"],
    risk_scores
)[0]


# ============================================================
# 9. Results
# ============================================================

print("\n========== RESULTS ==========")

print("Dataset: GBSG2")
print("Model: Classical RBF Survival SVM")
print("Features:", features)
print("Feature count:", len(features))

print("Alpha:", 100.0)
print("Gamma:", 0.001)
print("Rank ratio:", 1.0)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nC-index:", cindex)