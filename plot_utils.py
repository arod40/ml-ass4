import matplotlib.pyplot as plt


def plot_data(ax, X, y):
    y_ = y.squeeze()
    ax.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
    ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)

    ax.scatter(X[y_ == 1, 0], X[y_ == 1, 1], marker="o", color="blue")
    ax.scatter(X[y_ == -1, 0], X[y_ == -1, 1], marker="x", color="red")


def plot_consistency(axes, X_subset, y_subset, X, y, y1, y2):
    plot_data(axes[0], X, y)
    plot_data(axes[1], X_subset, y_subset)
    plot_data(axes[2], X, y1)
    plot_data(axes[3], X, y2)

    for ax in axes:
        ax.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
        ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)

