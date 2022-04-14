def plot_data(ax, X, y):
    y_ = y.squeeze()
    ax.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
    ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)

    ax.scatter(X[y_ == 1, 0], X[y_ == 1, 1], marker="o", color="blue")
    ax.scatter(X[y_ == -1, 0], X[y_ == -1, 1], marker="x", color="red")
