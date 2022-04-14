from numpy.random import randn, rand
import numpy as np


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

    return np.concatenate([x, y], axis=1)


def gen_sinus_uniform_data(n):
    x = rand(n, 1) * 5 - 2.5
    x.sort()
    f = (np.cos(x) + 2) / (np.cos(1.4 * x) + 2)
    noise = randn(n, 1) * f.std() * 0.5
    y = f + noise

    return np.concatenate([x, y], axis=1)
