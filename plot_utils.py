from cProfile import label
import numpy as np
from data_utils import build_nth_order_features, gen_sinus_normal_data


def plot_2D_function(ax, f, start, end, label="", color="blue"):
    x = np.linspace(start, end, 1000)
    y = [f(xi) for xi in x]
    ax.plot(x, y, color=color, label=label)


def plot_polynomial(ax, w, x, label="", color="blue"):
    n = w.shape[0]

    eval_nth_poly = lambda x, w: (build_nth_order_features(x, n - 1) @ w)
    y = eval_nth_poly(x, w)

    X = np.concatenate([x, y], axis=1)
    ax.plot(X[:, 0], X[:, 1], color=color, label=label)


def plot_data(ax, X, y, pos_label="positive", neg_label="negative"):
    y_ = y.squeeze()
    ax.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
    ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)

    ax.scatter(X[y_ == 1, 0], X[y_ == 1, 1], marker="o", color="blue", label=pos_label)
    ax.scatter(X[y_ == -1, 0], X[y_ == -1, 1], marker="x", color="red", label=neg_label)


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


def plot_error_bars(ax, results, color, label, move=-1):
    labels = results.keys()
    X = np.arange(len(labels))  # the label locations
    y = [results[key] for key in labels]

    width = 0.25
    ax.bar(X + move * width / 2, y, width, label=label, color=color)
    ax.set_xticks(X)
    ax.set_xticklabels(labels)


def plot_sparsity(ax, results):
    for _, weights, color in results:
        x = range(len(weights))
        y = abs(weights)
        ax.plot(x, y, color=color)
