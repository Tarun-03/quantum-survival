from sksurv.datasets import load_gbsg2
import numpy as np

# Load GBSG2 dataset
X, y = load_gbsg2()

print("========== DATASET ==========")

print("\nShape:")
print(X.shape)

print("\nColumns:")
print(X.columns.tolist())

print("\nData types:")
print(X.dtypes)

print("\nMissing values:")
print(X.isnull().sum())

print("\nFirst 5 rows of X:")
print(X.head())

print("\nTarget:")
print(y[:5])

print("\nTarget dtype:")
print(y.dtype)

print("\nEvent counts:")
print(np.unique(y["cens"], return_counts=True))

print("\n========== SURVIVAL INFORMATION ==========")

print("\nMinimum follow-up time:")
print(y["time"].min())

print("\nMaximum follow-up time:")
print(y["time"].max())

print("\nMean follow-up time:")
print(y["time"].mean())

print("\nMedian follow-up time:")
print(np.median(y["time"]))

print("\nEvent patients:")
print(y["cens"].sum())

print("\nCensored patients:")
print((~y["cens"]).sum())