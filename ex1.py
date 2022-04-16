from matplotlib.colors import rgb_to_hsv
from plot_utils import plot_data, plot_error_bars, plot_polynomial, plot_sparsity
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

from colorsys import hsv_to_rgb
from matplotlib.colors import to_rgb, to_hex

np_seed(8)
seed(8)


def evaluate_weights(X, y, w):
    y_predicted = X @ w
    return mean_squared_error(y, y_predicted)


def color_gradient(
    c1, c2, mix=0
):  # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1 = np.array(to_rgb(c1))
    c2 = np.array(to_rgb(c2))
    return to_hex((1 - mix) * c1 + mix * c2)


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
        lambdas = [10 ** i for i in range(-5, 5)]
        weights = []
        eins = {}
        eouts = {}
        for lamb in lambdas:
            w = ridge_regression(X_train, y_train, lamb)
            weights.append(w)
            eins[lamb] = evaluate_weights(X_train, y_train, w)
            eouts[lamb] = evaluate_weights(X_test, y_test, w)

        _, axes = plt.subplots(1, 2, figsize=(8, 4))

        plot_error_bars(axes[0], eouts, "blue", "Eout", move=0)
        plot_error_bars(axes[0], eins, "red", "Ein", move=0)

        axes[0].legend()
        axes[0].set_xlabel("lambda")
        axes[0].set_ylabel("mean square error")

        colors = [color_gradient("orange", "red", x / n) for x in range(n)]
        plot_sparsity(
            axes[1], zip(lambdas, weights, colors),
        )

        axes[1].set_xlabel("weight")
        axes[1].set_ylabel("absolute value")

        plt.show()
