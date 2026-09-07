import pandas as pd

from sksurv.datasets import load_gbsg2


# ============================================================
# Load dataset
# ============================================================

X, y = load_gbsg2()


# ============================================================
# Numerical features
# ============================================================

numerical_features = [
    "age",
    "estrec",
    "pnodes",
    "progrec",
    "tsize"
]


# ============================================================
# Create survival dataframe
# ============================================================

df = X[numerical_features].copy()

df["event"] = y["cens"]
df["time"] = y["time"]


# ============================================================
# Correlation with survival time
# ============================================================

print("========== CORRELATION WITH SURVIVAL TIME ==========")

print(
    df[numerical_features + ["time"]]
    .corr()["time"]
    .sort_values()
)


# ============================================================
# Correlation with event indicator
# ============================================================

print("\n========== CORRELATION WITH EVENT ==========")

print(
    df[numerical_features + ["event"]]
    .corr()["event"]
    .sort_values()
)


# ============================================================
# Basic statistics
# ============================================================

print("\n========== FEATURE STATISTICS ==========")

print(
    df[numerical_features].describe().T
)