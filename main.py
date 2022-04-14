from data_utils import gen_sinus_normal_data, gen_sinus_uniform_data
from plot_utils import plot_data
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    _, ax = plt.subplots(1, 2)
    y = np.ones((1000, 1))

    X = gen_sinus_normal_data(1000)
    plot_data(ax[0], X, y)

    X = gen_sinus_uniform_data(1000)
    plot_data(ax[1], X, y)

    plt.show()
