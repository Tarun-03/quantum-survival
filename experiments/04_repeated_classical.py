import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sksurv.datasets import load_gbsg2
from sksurv.svm import FastSurvivalSVM, FastKernelSurvivalSVM
from sksurv.metrics import concordance_index_censored


# ============================================================
# Load dataset
# ============================================================

X, y = load_gbsg2()


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
# Experiment settings
# ============================================================

seeds = [42, 43, 44, 45, 46]

linear_results = []
rbf_results = []


# ============================================================
# Run experiments
# ============================================================

for seed in seeds:

    print("\n========================================")
    print("Random seed:", seed)
    print("========================================")


    # --------------------------------------------------------
    # Train/Test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed
    )


    # --------------------------------------------------------
    # One-hot encoding
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Standardization
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_num = scaler.fit_transform(
        X_train[numerical_features]
    )

    X_test_num = scaler.transform(
        X_test[numerical_features]
    )


    # --------------------------------------------------------
    # Combine features
    # --------------------------------------------------------

    X_train_processed = np.hstack([
        X_train_num,
        X_train_cat
    ])

    X_test_processed = np.hstack([
        X_test_num,
        X_test_cat
    ])


    # ========================================================
    # LINEAR SURVIVAL SVM
    # ========================================================

    linear_model = FastSurvivalSVM(
        alpha=1.0,
        rank_ratio=1.0,
        max_iter=1000,
        tol=1e-5,
        random_state=seed
    )

    linear_model.fit(
        X_train_processed,
        y_train
    )

    linear_prediction = linear_model.predict(
        X_test_processed
    )

    linear_cindex = concordance_index_censored(
        y_test["cens"],
        y_test["time"],
        linear_prediction
    )[0]

    linear_results.append(linear_cindex)


    # ========================================================
    # RBF SURVIVAL SVM
    # ========================================================

    rbf_model = FastKernelSurvivalSVM(
        alpha=100.0,
        rank_ratio=1.0,
        kernel="rbf",
        gamma=0.001,
        max_iter=1000,
        tol=1e-5,
        random_state=seed
    )

    rbf_model.fit(
        X_train_processed,
        y_train
    )

    rbf_prediction = rbf_model.predict(
        X_test_processed
    )

    rbf_cindex = concordance_index_censored(
        y_test["cens"],
        y_test["time"],
        rbf_prediction
    )[0]

    rbf_results.append(rbf_cindex)


    # ========================================================
    # Print current results
    # ========================================================

    print("Linear C-index:", linear_cindex)
    print("RBF C-index:", rbf_cindex)


# ============================================================
# Summary
# ============================================================

print("\n\n========================================")
print("FINAL SUMMARY")
print("========================================")

print("\nLinear Survival SVM")
print("-------------------")

print("Results:", linear_results)
print("Mean:", np.mean(linear_results))
print("Std:", np.std(linear_results))


print("\nRBF Survival SVM")
print("----------------")

print("Results:", rbf_results)
print("Mean:", np.mean(rbf_results))
print("Std:", np.std(rbf_results))