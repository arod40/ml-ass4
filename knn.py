from math import sqrt
from random import choice, sample, seed

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import random
from numpy.random import seed as np_seed
from tqdm import tqdm

from data_utils import gen_linear_data
from plot_utils import plot_consistency, plot_data

np_seed(9)
seed(9)


def euclidean_distance(x1, x2):
    return sqrt(((x1 - x2) ** 2).sum())


def get_k_nearest_neighbors(X, y, p, k, subject_to=None):
    dist = sorted(
        [
            (euclidean_distance(p, data_point), i)
            for i, data_point in enumerate(X)
            if subject_to is None or subject_to(data_point, i)
        ]
    )
    k_near_idxs = [idx for _, idx in dist[:k]]
    return X[k_near_idxs], y[k_near_idxs], k_near_idxs


class KNN:
    def __init__(self, k):
        assert k % 2 == 1, "The value of k must be odd"
        self.k = k
        self.data = None
        self.condensed_data = None
        self._train_predict = None

    @property
    def _self_predict(self):
        if self._train_predict is None:
            X, _ = self.data
            self._train_predict = self.predict_batch(X)
        return self._train_predict

    @property
    def is_fit(self):
        return self.data is not None

    @property
    def is_condensed(self):
        return self.condensed_data is not None

    def fit(self, X, y):
        self.data = X, y

    def predict(self, p, data=None):
        assert (
            data is not None or self.is_fit
        ), "If model is not fit, predict should receive some data"

        X, y = (
            data
            or (self.condensed_data if self.is_condensed else None)
            or (self.data if self.is_fit else None)
        )
        _, labels, _ = get_k_nearest_neighbors(X, y, p, self.k)
        return 1 if labels.mean() > 0 else -1

    def predict_batch(self, P, data=None, show_progress=False):
        return np.array(
            [self.predict(p, data) for p in (tqdm(P) if show_progress else P)]
        ).reshape((-1, 1))

    def condense(self, verbose=False, interactive=False):
        if verbose:
            print("Condensing the training data")

        X, y = self.data
        N = X.shape[0]

        chosen = np.array([False] * N)

        # choose k random initial points
        chosen[sample(range(N), self.k)] = True

        if interactive:
            plt.ion()
            fig, axes = plt.subplots(2, 2, figsize=(8, 8))

        # loop until the subset is train consistent
        i = 0
        while True:
            subset_predict = self.predict_batch(X, (X[chosen], y[chosen]))
            consistent = (self._self_predict == subset_predict).squeeze(1)

            if verbose and i % int(0.02 * N) == 0:
                print(f"Subset size: {chosen.sum()}")
                print(f"Inconsistent: {(~consistent).sum()}")
            i += 1

            if interactive:
                for r in axes:
                    for ax in r:
                        ax.clear()
                plot_consistency(
                    axes, X[chosen], y[chosen], X, y, self._self_predict, subset_predict
                )
                fig.canvas.draw()
                fig.canvas.flush_events()
                input("Press a key to continue")

            if consistent.all():
                break

            # choose a point to extend S
            rand_inc = choice(range(N - consistent.sum()))
            x_star = X[~consistent][rand_inc]  # inconsistent point
            gD_x_star = self._self_predict[~consistent][
                rand_inc
            ]  # class of the inconsistent point according to train data

            # get closest neighbor not chosen and with same class
            _, _, idxs = get_k_nearest_neighbors(
                X,
                y,
                x_star,
                1,
                subject_to=lambda point, idx: not chosen[idx] and y[idx] == gD_x_star,
            )

            # add that point to S
            chosen[idxs[0]] = True

        self.condensed_data = X[chosen], y[chosen]

        if verbose:
            print(f"Kept {chosen.sum()*100/N}% of the data")


if __name__ == "__main__":
    knn = KNN(3)
    c = 3 * np.random.randn(2, 2)
    X, y = gen_linear_data(c, 1000)
    knn.fit(X, y)

    import sys

    test = sys.argv[1]

    if test == "plot":
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

    elif test == "condense":
        knn.condense(verbose=True, interactive=True)

