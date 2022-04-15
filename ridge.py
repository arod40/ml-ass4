import numpy as np


def ridge_regression(X, y, lamb):
    X_t = X.transpose()
    return np.linalg.inv(X_t @ X + lamb * np.identity(X_t.shape[0])) @ X_t @ y
