from data_utils import gen_linear_data, gen_sinus_normal_data, gen_sinus_uniform_data
from plot_utils import plot_data
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    _, ax = plt.subplots(1, 3)
    y = np.ones((1000, 1))

    X = gen_sinus_normal_data(1000)
    plot_data(ax[0], X, y)

    X = gen_sinus_uniform_data(1000)
    plot_data(ax[1], X, y)

    c = 3 * np.random.randn(2, 2)
    X, y = gen_linear_data(c, 1000)
    plot_data(ax[2], X, y)

    plt.show()
