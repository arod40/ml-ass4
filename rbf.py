from math import e

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

from data_utils import gen_sinus_normal_data
from knn import euclidean_distance
from plot_utils import plot_2D_function, plot_data


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


class RBFNetwork:
    def __init__(self, kernel, r, k, use_kmeans=False):
        self.kernel = kernel
        self.r = r
        self.k = k
        self.use_kmeans = use_kmeans
        self.kmeans = KMeans(n_clusters=k)

    def _pseudo_inverse(self, X, y):
        X_t = X.transpose()
        return np.linalg.inv(X_t @ X) @ X_t @ y

    def _transform_nonlinear(self, X):
        return np.concatenate(
            [
                self.kernel(euclidean_distance(X, mu.reshape(1, -1)) / self.r)
                for mu in self.centers
            ],
            axis=1,
        )

    def _expand_for_bias(self, X):
        return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

    def fit(self, X, y):
        if self.use_kmeans:
            self.kmeans.fit(X)
            self.centers = self.kmeans.cluster_centers_
        else:
            self.centers = np.linspace(X.min(axis=0), X.max(axis=0), self.k)
        X_transform = self._expand_for_bias(self._transform_nonlinear(X))
        self.weights = self._pseudo_inverse(X_transform, y)

    def predict(self, X):
        X_transform = self._expand_for_bias(self._transform_nonlinear(X))
        return X_transform @ self.weights


if __name__ == "__main__":
    x, y = gen_sinus_normal_data(1000)

    npRBF = NonParametricRBF(gaussian_kernel, 0.3)
    npRBF.fit(x, y)

    pRBF = RBFNetwork(gaussian_kernel, 0.3, 5, use_kmeans=True)
    pRBF.fit(x, y)

    _, ax = plt.subplots(1, 1)

    plot_data(
        ax,
        np.concatenate([x, y], axis=1),
        np.ones(x.shape),
        pos_label="train",
        neg_label=None,
    )

    f = lambda x: npRBF.predict(np.array([[x]])).squeeze()
    g = lambda x: pRBF.predict(np.array([[x]])).squeeze()
    plot_2D_function(ax, f, x.min(), x.max(), label="non-parametric rbf", color="red")
    plot_2D_function(ax, g, x.min(), x.max(), label="parametric rbf", color="green")

    ax.vlines(
        pRBF.centers.squeeze(),
        y.min(),
        y.max(),
        label="parametric centers",
        color="gray",
    )

    ax.legend()
    plt.show()

