from math import sqrt
import numpy as np
from numpy.random import random
import matplotlib.pyplot as plt

from data_utils import gen_linear_data
from plot_utils import plot_data

np.random.seed(0)


def euclidean_distance(x1, x2):
    return sqrt(((x1 - x2) ** 2).sum())


def naive_k_nearest_neigbors(X, y, p, k):
    dist = sorted(
        [(euclidean_distance(p, data_point), i) for i, data_point in enumerate(X)]
    )
    k_near_idxs = [idx for _, idx in dist[:k]]
    return X[k_near_idxs], y[k_near_idxs]


class KNN:
    def __init__(self, k):
        assert k % 2 == 1, "The value of k must be odd"
        self.k = k
        self.data = None

    def fit(self, X, y):
        self.data = X, y

    def predict(self, p):
        assert self.data, "Model has not been fit"
        X, y = self.data
        _, labels = naive_k_nearest_neigbors(X, y, p, self.k)
        return 1 if labels.mean() > 0.5 else 0


if __name__ == "__main__":
    knn = KNN(9)
    c = 3 * np.random.randn(2, 2)
    X, y = gen_linear_data(c, 1000)
    knn.fit(X, y)

    bottom, top, left, right = (
        X[:, 0].min() + 1,
        X[:, 0].max() - 1,
        X[:, 1].min() + 1,
        X[:, 1].max() - 1,
    )
    width = right - left
    height = top - bottom

    while True:
        p = np.array([random() * width + left, random() * height + bottom])
        print(p)
        label = knn.predict(p)

        _, ax = plt.subplots(1, 1)
        plot_data(ax, X, y)
        plt.scatter(
            [p[0]], [p[1]], marker="o", color="purple" if label == 1 else "orange"
        )

        plt.show()
