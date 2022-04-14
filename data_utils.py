from numpy.random import randn, rand
import numpy as np


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
