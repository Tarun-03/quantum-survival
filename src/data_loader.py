from sksurv.datasets import load_gbsg2
import numpy as np


def load_dataset():
    X, y = load_gbsg2()
    return X, y


if __name__ == "__main__":
    X, y = load_dataset()

    print("Features:")
    print(X.head())

    print("\nShape:")
    print(X.shape)

    print("\nFeature types:")
    print(X.dtypes)

    print("\nTarget:")
    print(y[:5])

    print("\nTarget dtype:")
    print(y.dtype)

    print("\nEvent count:")
    print(np.unique(y["cens"], return_counts=True))