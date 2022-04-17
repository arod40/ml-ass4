import numpy as np
from numpy.random import seed as np_seed
from random import seed
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from data_utils import gen_sinus_normal_data
from plot_utils import plot_2D_function, plot_data
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


if __name__ == "__main__":
    N = 100
    x_train, y_train = gen_sinus_normal_data(N)
    x_test, y_test = gen_sinus_normal_data(10 * N)

    _, ax = plt.subplots(1, 1)

    plot_data(
        ax,
        np.concatenate(
            [
                np.concatenate([x_train, y_train], axis=1),
                np.concatenate([x_test, y_test], axis=1),
            ],
            axis=0,
        ),
        np.concatenate([np.ones(x_train.shape), -np.ones(x_test.shape)], axis=0),
        pos_label="train",
        neg_label="test",
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

    Ein, Eout, f = experiment(x_train, y_train, x_test, y_test, True, 0.3, 5)
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
