from plot_utils import plot_data, plot_polynomial
from ridge import ridge_regression
from data_utils import (
    build_nth_order_features,
    gen_sinus_uniform_data,
)
import numpy as np
from numpy.random import seed as np_seed
from random import seed
import matplotlib.pyplot as plt

np_seed(8)
seed(8)

if __name__ == "__main__":
    import sys

    item = sys.argv[1]

    if item == "2":
        N = 20
        lamb = 10
        x_train, y_train = gen_sinus_uniform_data(N)
        x_test, y_test = gen_sinus_uniform_data(10 * N)

        left = min(x_train.min(), x_test.min())
        right = max(x_train.max(), x_test.max())

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

        n = 10
        X_train = build_nth_order_features(x_train, n)
        w = ridge_regression(X_train, y_train, lamb)

        x = np.empty((100, 1))
        x[:, 0] = np.linspace(left, right, 100)
        plot_polynomial(ax, w, x, label=f"n={n}", color="green")

        ax.legend()
        plt.show()

    if item == "3":
        pass
