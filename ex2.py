from more_itertools import sample
import numpy as np
from numpy.random import seed as np_seed
from random import seed
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from data_utils import gen_sinus_normal_data, gen_sinus_uniform_data
from plot_utils import (
    color_gradient,
    plot_2D_function,
    plot_data,
    plot_error_bars,
    plot_functions_colors,
)
from rbf import NonParametricRBF, RBFNetwork, gaussian_kernel

np_seed(8)
seed(8)


def evaluate(rbf, X, y):
    y_predicted = rbf.predict_batch(X)
    return mean_squared_error(y, y_predicted)


def experiment(X_train, y_train, X_test, y_test, parametric, r, k=None):
    assert not parametric or k is not None

    if parametric:
        model = RBFNetwork(gaussian_kernel, r, k)
    else:
        model = NonParametricRBF(gaussian_kernel, r)

    model.fit(X_train, y_train)

    ein = evaluate(model, X_train, y_train)
    eout = evaluate(model, X_test, y_test)
    f = lambda x: model.predict(np.array([[x]])).squeeze()

    return ein, eout, f


def vary_radius(values_of_r, parametric, k, x_train, y_train, x_test, y_test):
    N = len(x_train)
    colors = [
        color_gradient("orange", "red", x / len(values_of_r))
        for x in range(len(values_of_r))
    ]

    eins = {}
    eouts = {}
    functions = []
    for r in values_of_r:
        ein, eout, f = experiment(x_train, y_train, x_test, y_test, parametric, r, k)
        eins[r] = ein
        eouts[r] = eout
        functions.append(f)

    _, axes = plt.subplots(1, 2, figsize=(8, 4))

    plot_error_bars(axes[0], eouts, "blue", "Eout", move=0)
    plot_error_bars(axes[0], eins, "red", "Ein", move=0)
    axes[0].set_xlabel("radius")
    axes[0].set_ylabel("mean squared error")
    axes[0].legend()

    plot_functions_colors(
        axes[1],
        values_of_r,
        "radius",
        functions,
        colors,
        min(x_train.min(), x_test.min()),
        max(x_train.max(), x_test.max()),
    )

    plot_data(
        axes[1],
        np.concatenate([x_train, y_train], axis=1),
        np.ones((N, 1)),
        pos_label="train data",
        neg_label="",
    )
    plt.show()


def vary_centers(values_of_k, r, x_train, y_train, x_test, y_test):
    N = len(x_train)
    colors = [
        color_gradient("orange", "red", x / len(values_of_k))
        for x in range(len(values_of_k))
    ]

    eins = {}
    eouts = {}
    functions = []
    for k in values_of_k:
        ein, eout, f = experiment(x_train, y_train, x_test, y_test, True, r, k)
        eins[k] = ein
        eouts[k] = eout
        functions.append(f)

    _, axes = plt.subplots(1, 2, figsize=(8, 4))

    plot_error_bars(axes[0], eouts, "blue", "Eout", move=0)
    plot_error_bars(axes[0], eins, "red", "Ein", move=0)
    axes[0].set_xlabel("number of centers")
    axes[0].set_ylabel("mean squared error")
    axes[0].legend()

    plot_functions_colors(
        axes[1],
        values_of_k,
        "number of centers",
        functions,
        colors,
        min(x_train.min(), x_test.min()),
        max(x_train.max(), x_test.max()),
    )

    plot_data(
        axes[1],
        np.concatenate([x_train, y_train], axis=1),
        np.ones((N, 1)),
        pos_label="train data",
        neg_label="",
    )
    plt.show()


def vary_both(values_of_k, values_of_r, x_train, y_train, x_test, y_test):
    N = len(x_train)
    n = len(values_of_k) * len(values_of_r)
    colors = [color_gradient("orange", "red", x / n) for x in range(n)]

    R = x_train.max() - x_train.min()

    eins = []
    eouts = []
    functions = []
    for k in values_of_k:
        for r in values_of_r:
            ein, eout, f = experiment(x_train, y_train, x_test, y_test, True, r, k)
            value = abs(R / k - r)
            eins.append((value, ein))
            eouts.append((value, eout))
            functions.append((value, f))

    eins.sort()
    eouts.sort()
    functions.sort()

    _, axes = plt.subplots(1, 2, figsize=(8, 4))

    plot_error_bars(axes[0], dict(eouts), "blue", "Eout", move=0)
    plot_error_bars(axes[0], dict(eins), "red", "Ein", move=0)
    axes[0].set_ylim(0, 0.040)
    axes[0].set_xlabel("|R/k-r|")
    axes[0].set_ylabel("mean squared error")
    axes[0].legend()

    values = [v for v, _ in functions]
    functions = [f for _, f in functions]
    plot_functions_colors(
        axes[1],
        values,
        "|R/k-r|",
        functions,
        colors,
        min(x_train.min(), x_test.min()),
        max(x_train.max(), x_test.max()),
    )

    plot_data(
        axes[1],
        np.concatenate([x_train, y_train], axis=1),
        np.ones((N, 1)),
        pos_label="train data",
        neg_label="",
    )
    plt.show()


if __name__ == "__main__":
    N = 20
    x_train, y_train = gen_sinus_uniform_data(N)
    x_test, y_test = gen_sinus_uniform_data(10 * N)

    import sys

    item = sys.argv[1]

    if item == "plot":
        _, ax = plt.subplots(1, 1, figsize=(4, 4))

        plot_data(
            ax,
            np.concatenate([x_train, y_train], axis=1),
            np.ones((N, 1)),
            pos_label="train data",
            neg_label="",
        )

        Ein, Eout, f = experiment(x_train, y_train, x_test, y_test, False, 0.3)
        print("Non-parametric")
        print("Ein:", Ein, "Eout:", Eout)
        plot_2D_function(
            ax,
            f,
            min(x_train.min(), x_test.min()),
            max(x_train.max(), x_test.max()),
            label="non-parametric rbf",
            color="orange",
        )

        Ein, Eout, f = experiment(x_train, y_train, x_test, y_test, True, 0.7, 5)
        print("Parametric")
        print("Ein:", Ein, "Eout:", Eout)
        plot_2D_function(
            ax,
            f,
            min(x_train.min(), x_test.min()),
            max(x_train.max(), x_test.max()),
            label="parametric rbf",
            color="green",
        )

        ax.legend()
        plt.show()

    if item == "r":
        k = 5
        values_of_r = np.linspace(0.1, 1, 30)

        vary_radius(values_of_r, False, k, x_train, y_train, x_test, y_test)
        vary_radius(values_of_r, True, k, x_train, y_train, x_test, y_test)

    if item == "k":
        r = 0.7
        values_of_k = np.array(range(1, 11))

        vary_centers(values_of_k, r, x_train, y_train, x_test, y_test)

    if item == "both":
        values_of_r = np.linspace(0.1, 1, 30)
        values_of_k = np.array(range(1, 11))

        vary_both(values_of_k, values_of_r, x_train, y_train, x_test, y_test)
