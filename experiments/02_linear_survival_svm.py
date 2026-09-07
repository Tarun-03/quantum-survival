import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sksurv.datasets import load_gbsg2
from sksurv.svm import FastSurvivalSVM
from sksurv.metrics import concordance_index_censored


# ============================================================
# 1. Load dataset
# ============================================================

X, y = load_gbsg2()


# ============================================================
# 2. Define feature types
# ============================================================

numerical_features = [
    "age",
    "estrec",
    "pnodes",
    "progrec",
    "tsize"
]

categorical_features = [
    "horTh",
    "menostat",
    "tgrade"
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
# 4. One-hot encode categorical features
# ============================================================

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

X_train_cat = encoder.fit_transform(
    X_train[categorical_features]
)

X_test_cat = encoder.transform(
    X_test[categorical_features]
)


# ============================================================
# 5. Standardize numerical features
# ============================================================

scaler = StandardScaler()

X_train_num = scaler.fit_transform(
    X_train[numerical_features]
)

X_test_num = scaler.transform(
    X_test[numerical_features]
)


# ============================================================
# 6. Combine features
# ============================================================

X_train_processed = np.hstack([
    X_train_num,
    X_train_cat
])

X_test_processed = np.hstack([
    X_test_num,
    X_test_cat
])


# ============================================================
# 7. Create Linear Survival SVM
# ============================================================

model = FastSurvivalSVM(
    alpha=1.0,
    rank_ratio=1.0,
    max_iter=1000,
    tol=1e-5,
    random_state=42
)


# ============================================================
# 8. Train
# ============================================================

print("Training Linear Survival SVM...")

model.fit(
    X_train_processed,
    y_train
)


# ============================================================
# 9. Generate risk scores
# ============================================================

prediction = model.predict(
    X_test_processed
)


# ============================================================
# 10. Calculate C-index
# ============================================================

c_index = concordance_index_censored(
    y_test["cens"],
    y_test["time"],
    prediction
)[0]


# ============================================================
# 11. Results
# ============================================================

print("\n========== RESULTS ==========")

print("Dataset: GBSG2")
print("Model: Linear Survival SVM")
print("Alpha:", 1.0)
print("Rank ratio:", 1.0)

print("\nTraining samples:", X_train_processed.shape[0])
print("Testing samples:", X_test_processed.shape[0])
print("Features:", X_train_processed.shape[1])

print("\nC-index:", c_index)