from random import seed

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import seed as np_seed
from sklearn.metrics import mean_squared_error

from data_utils import (
    build_nth_order_features,
    gen_sinus_normal_data,
    gen_sinus_uniform_data,
)
from plot_utils import (
    color_gradient,
    plot_data,
    plot_error_bars,
    plot_polynomial,
    plot_sparsity,
)
from ridge import ridge_regression

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
        _, ax = plt.subplots(1, 1, figsize=(4, 4))

        plot_data(
            ax,
            np.concatenate([x_train, y_train], axis=1),
            np.ones((N, 1)),
            pos_label="train data",
            neg_label="",
        )
        x = np.empty((100, 1))
        x[:, 0] = np.linspace(left, right, 100)

        w = ridge_regression(X_train, y_train, 10)
        plot_polynomial(ax, w, x, color="green", label="with regularization")

        w = ridge_regression(X_train, y_train, 0)
        plot_polynomial(ax, w, x, color="red", label="without regularization")

        ax.legend()
        plt.show()

    if item == "3":
        lambdas = 10 ** np.linspace(-2, 1.5, 50)
        weights = []
        eins = {}
        eouts = {}
        for lamb in lambdas:
            w = ridge_regression(X_train, y_train, lamb)
            weights.append(w)
            eins[lamb] = evaluate_weights(X_train, y_train, w)
            eouts[lamb] = evaluate_weights(X_test, y_test, w)

        _, axes = plt.subplots(1, 2, figsize=(8, 4))

        # Plot error as function of lambda
        plot_error_bars(axes[0], eouts, "blue", "Eout", move=0)
        plot_error_bars(axes[0], eins, "red", "Ein", move=0)

        axes[0].legend()
        axes[0].set_xlabel("lambda")
        axes[0].set_ylabel("mean squared error")

        # Plot scarcity
        colors = [
            color_gradient("orange", "red", x / len(lambdas))
            for x in range(len(lambdas))
        ]
        plot_sparsity(
            axes[1], lambdas, weights, colors,
        )

        axes[1].set_xlabel("weight")
        axes[1].set_ylabel("absolute value")

        plt.show()
