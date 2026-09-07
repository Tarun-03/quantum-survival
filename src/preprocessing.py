from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sksurv.datasets import load_gbsg2


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

X, y = load_gbsg2()


# --------------------------------------------------
# 2. Define feature types
# --------------------------------------------------

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


# --------------------------------------------------
# 3. Train/Test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# --------------------------------------------------
# 4. Encode categorical features
# --------------------------------------------------

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


# --------------------------------------------------
# 5. Scale numerical features
# --------------------------------------------------

scaler = StandardScaler()

X_train_num = scaler.fit_transform(
    X_train[numerical_features]
)

X_test_num = scaler.transform(
    X_test[numerical_features]
)


# --------------------------------------------------
# 6. Combine numerical + categorical features
# --------------------------------------------------

import numpy as np

X_train_processed = np.hstack([
    X_train_num,
    X_train_cat
])

X_test_processed = np.hstack([
    X_test_num,
    X_test_cat
])


# --------------------------------------------------
# 7. Print results
# --------------------------------------------------

print("\nOriginal feature count:")
print(X.shape[1])

print("\nProcessed training shape:")
print(X_train_processed.shape)

print("\nProcessed testing shape:")
print(X_test_processed.shape)

print("\nTraining event count:")
print(y_train["cens"].sum())

print("\nTesting event count:")
print(y_test["cens"].sum())