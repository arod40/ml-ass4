from pathlib import Path

import numpy as np
from numpy.random import rand, randn


def gen_linear_data(c, n):
    assert n % 2 == 0

    y = np.concatenate([-1 * np.ones((n // 2, 1)), np.ones((n // 2, 1))])
    X = np.random.randn(n, 2) + c[((y.squeeze() + 3) / 2 - 1).astype("int64")]

    return X, y


def gen_sinus_normal_data(n):
    x = randn(n, 1) * 2
    x.sort()
    f = (np.cos(x) + 2) / (np.cos(1.4 * x) + 2)
    noise = randn(n, 1) * f.std() * 0.5
    y = f + noise

    return x, y


def gen_sinus_uniform_data(n):
    x = rand(n, 1) * 5 - 2.5
    x.sort()
    f = (np.cos(x) + 2) / (np.cos(1.4 * x) + 2)
    noise = randn(n, 1) * f.std() * 0.5
    y = f + noise

    return x, y


def read_digits_data(file: Path):
    train = np.array(
        [
            list(map(float, line.strip().split()))
            for line in file.read_text().splitlines()
        ]
    )

    X, y = train[:, 1:], (train[:, :1] == 1) * 2 - 1
    return X, y


def build_nth_order_features(x, n):
    return np.concatenate([x ** i for i in range(n + 1)], axis=1)


if __name__ == "__main__":
    import sys

    import matplotlib.pyplot as plt

    from plot_utils import plot_data

    item = sys.argv[1]

    if item == "digits":
        X, y = read_digits_data(Path("./data/features.train.txt"))

        _, ax = plt.subplots(1, 1)

        plot_data(ax, X, y)
        plt.show()

    if item == "sinus":
        N = 50
        _, ax = plt.subplots(1, 2, figsize=(8, 4))
        y_ = np.ones((N, 1))

        x, y = gen_sinus_normal_data(N)
        X = np.concatenate([x, y], axis=1)
        plot_data(ax[0], X, y_)

        x, y = gen_sinus_uniform_data(N)
        X = np.concatenate([x, y], axis=1)
        plot_data(ax[1], X, y_)

        plt.show()

    if item == "poly":
        N = 5
        x, y = gen_sinus_normal_data(N)
        print(x)
        print(build_nth_order_features(x, 2))
