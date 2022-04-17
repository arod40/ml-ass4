from math import e
import numpy as np
from tqdm import tqdm
from knn import euclidean_distance

from data_utils import gen_sinus_normal_data
from plot_utils import plot_2D_function, plot_data
import matplotlib.pyplot as plt


def gaussian_kernel(x):
    return e ** -(0.5 * x ** 2)


def euclidean_distance(x1, x2):
    return np.sqrt(((x1 - x2) ** 2).sum(axis=1, keepdims=True))


class NonParametricRBF:
    def __init__(self, kernel, r):
        self.kernel = kernel
        self.r = r

    def fit(self, X, y):
        self.data = X, y

    def predict(self, x):
        X_train, y_train = self.data
        alphas = self.kernel(euclidean_distance(X_train, x) / self.r)
        return np.transpose((alphas / alphas.sum())) @ y_train

    def predict_batch(self, X, show_progress=False):
        return np.array(
            [self.predict(x) for x in (tqdm(X) if show_progress else X)]
        ).reshape((-1, 1))


if __name__ == "__main__":
    x, y = gen_sinus_normal_data(50)

    npRBF = NonParametricRBF(gaussian_kernel, 1)
    npRBF.fit(x, y)

    _, ax = plt.subplots(1, 1)

    plot_data(
        ax,
        np.concatenate([x, y], axis=1),
        np.ones(x.shape),
        pos_label="train",
        neg_label=None,
    )

    f = lambda x: npRBF.predict(np.array([[x]])).squeeze()
    plot_2D_function(ax, f, x.min(), x.max(), label="non-parametric rbf", color="red")
    ax.legend()
    plt.show()

