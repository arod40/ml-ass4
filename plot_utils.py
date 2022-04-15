import matplotlib.pyplot as plt


def plot_data(ax, X, y):
    y_ = y.squeeze()
    ax.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
    ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)

    ax.scatter(X[y_ == 1, 0], X[y_ == 1, 1], marker="o", color="blue")
    ax.scatter(X[y_ == -1, 0], X[y_ == -1, 1], marker="x", color="red")


def plot_consistency(axes, X_subset, y_subset, X, y, y1, y2):
    plot_data(axes[0, 0], X, y)
    axes[0, 0].set_title("Dataset")
    plot_data(axes[0, 1], X, y1)
    axes[0, 1].set_title("Dataset predictions")
    plot_data(axes[1, 0], X_subset, y_subset)
    axes[1, 0].set_title("Subset data")
    plot_data(axes[1, 1], X, y2)
    axes[1, 1].set_title("Subset data predictions")

    for r in axes:
        for ax in r:
            ax.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
            ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)


def plot_train_and_test(
    axes, X_train, y_train, y_predict_train, X_test, y_test, y_predict_test
):
    plot_data(axes[0, 0], X_train, y_train)
    axes[0, 0].set_title("Train data")
    plot_data(axes[0, 1], X_train, y_predict_train)
    axes[0, 1].set_title("Prediction on train data")
    plot_data(axes[1, 0], X_test, y_test)
    axes[1, 0].set_title("Test data")
    plot_data(axes[1, 1], X_test, y_predict_test)
    axes[1, 1].set_title("Prediction on test data")

    for r in axes:
        for ax in r:
            ax.set_xlim(
                min(X_train[:, 0].min(), X_test[:, 0].min()) - 1,
                max(X_train[:, 0].max(), X_test[:, 0].max()) + 1,
            )
            ax.set_ylim(
                min(X_train[:, 1].min(), X_test[:, 1].min()) - 1,
                max(X_train[:, 1].max(), X_test[:, 1].max()) + 1,
            )

