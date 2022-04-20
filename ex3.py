import sys
from pathlib import Path
from random import seed

import matplotlib.pyplot as plt
from numpy.random import seed as np_seed
from path import Path
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from data_utils import read_digits_data
from knn import KNN
from plot_utils import plot_train_and_test
from datetime import datetime

import numpy as np

random_state = 0
np_seed(9)
seed(9)


def evaluate(knn, X_train, X_test, y_train, y_test, verbose=False, interactive=False):
    if verbose:
        print("Predicting on train data")
    y_predict_train = knn.predict_batch(X_train, show_progress=verbose)
    Ein = 1 - accuracy_score(y_train, y_predict_train)

    if verbose:
        print("Predicting on test data")
    y_predict_test = knn.predict_batch(X_test, show_progress=verbose)
    Eout = 1 - accuracy_score(y_test, y_predict_test)

    if verbose:
        print("Ein:", Ein, "Eout:", Eout)

    if interactive:
        _, axes = plt.subplots(2, 2, figsize=(8, 8))
        plot_train_and_test(
            axes, X_train, y_train, y_predict_train, X_test, y_test, y_predict_test
        )
        plt.show()

    return Ein, Eout


def experiment(X, y, verbose=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=500)
    knn = KNN(3)
    knn.fit(X_train, y_train)

    t0 = datetime.now()
    Ein, Eout = evaluate(
        knn, X_train, X_test, y_train, y_test, verbose=False, interactive=False
    )
    t = (datetime.now() - t0).total_seconds()

    t0 = datetime.now()
    knn.condense(verbose=False)
    t_condense = (datetime.now() - t0).total_seconds()

    t0 = datetime.now()
    Ein_condensed, Eout_condensed = evaluate(
        knn, X_train, X_test, y_train, y_test, verbose=False, interactive=False
    )
    t_condensed = (datetime.now() - t0).total_seconds()

    if verbose:
        print("Full training data...")
        print("Ein:", Ein, "Eout:", Eout)

        print("Condensed data...")
        print("Ein:", Ein_condensed, "Eout:", Eout_condensed)

    return Ein, Eout, Ein_condensed, Eout_condensed, t, t_condense, t_condensed


if __name__ == "__main__":
    item = sys.argv[1]

    train_digits_file = Path("./data/features.train.txt")
    test_digits_file = Path("./data/features.test.txt")
    X_train, y_train = read_digits_data(train_digits_file)
    X_test, y_test = read_digits_data(test_digits_file)
    X, y = np.concatenate([X_train, X_test]), np.concatenate([y_train, y_test])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=500, random_state=random_state
    )

    knn = KNN(3)
    knn.fit(X_train, y_train)

    if item == "b":
        Ein, Eout = evaluate(
            knn, X_train, X_test, y_train, y_test, verbose=True, interactive=True
        )

    if item == "c":
        knn.condense(verbose=True)
        Ein, Eout = evaluate(
            knn, X_train, X_test, y_train, y_test, verbose=True, interactive=True
        )

    if item == "d":
        n = 1000
        Eins = []
        Eouts = []
        Eins_condensed = []
        Eouts_condensed = []
        times = []
        times_condense = []
        times_condensed = []

        for _ in tqdm(range(n)):
            (
                Ein,
                Eout,
                Ein_condensed,
                Eout_condensed,
                t,
                t_condense,
                t_condensed,
            ) = experiment(X, y, verbose=True)
            Eins.append(Ein)
            Eouts.append(Eout)
            Eins_condensed.append(Ein_condensed)
            Eouts_condensed.append(Eout_condensed)
            times.append(t)
            times_condense.append(t_condense)
            times_condensed.append(t_condensed)

        print("AVERAGE RESULTS")

        print("Full training data...")
        print(
            "Ein:", np.mean(Eins), "Eout:", np.mean(Eouts), "Avg. time:", np.mean(times)
        )

        print("Condensed data...")
        print(
            "Ein:",
            np.mean(Eins_condensed),
            "Eout:",
            np.mean(Eouts_condensed),
            "Avg. time (to condense):",
            np.mean(times_condense),
            "Avg. time (to predict):",
            np.mean(times_condensed),
        )

