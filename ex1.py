from plot_utils import plot_data, plot_error_bars, plot_polynomial
from ridge import ridge_regression
from data_utils import (
    build_nth_order_features,
    gen_sinus_uniform_data,
)
import numpy as np
from numpy.random import seed as np_seed
from random import seed
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

np_seed(8)
seed(8)


def evaluate_weights(X, y, w):
    y_predicted = X @ w
    return mean_squared_error(y, y_predicted)


if __name__ == "__main__":
    import sys

    item = sys.argv[1]

    N = 20
    n = 10
    x_train, y_train = gen_sinus_uniform_data(N)
    x_test, y_test = gen_sinus_uniform_data(10 * N)
    left = min(x_train.min(), x_test.min())
    right = max(x_train.max(), x_test.max())
    X_train = build_nth_order_features(x_train, n)
    X_test = build_nth_order_features(x_test, n)

    if item == "2":
        lamb = 10

        _, ax = plt.subplots(1, 1, figsize=(4, 4))

        plot_data(
            ax,
            np.concatenate(
                [
                    np.concatenate([x_test, y_test], axis=1),
                    np.concatenate([x_train, y_train], axis=1),
                ],
                axis=0,
            ),
            np.concatenate([-np.ones((10 * N, 1)), np.ones((N, 1))], axis=0),
            pos_label="train data",
            neg_label="test data",
        )

        w = ridge_regression(X_train, y_train, lamb)

        x = np.empty((100, 1))
        x[:, 0] = np.linspace(left, right, 100)
        plot_polynomial(ax, w, x, label=f"n={n}", color="green")

        ax.legend()
        plt.show()

    if item == "3":
        weights = {}
        eins = {}
        eouts = {}
        for lamb in [10 ** i for i in range(-5, 5)]:
            w = ridge_regression(X_train, y_train, lamb)
            weights[lamb] = w
            eins[lamb] = evaluate_weights(X_train, y_train, w)
            eouts[lamb] = evaluate_weights(X_test, y_test, w)

        _, ax = plt.subplots(1, 1, figsize=(4, 4))

        plot_error_bars(ax, eouts, "blue", "Eout", move=0)
        plot_error_bars(ax, eins, "red", "Ein", move=0)

        ax.legend()
        ax.set_xlabel("lambda")
        ax.set_ylabel("mean square error")
        plt.show()
