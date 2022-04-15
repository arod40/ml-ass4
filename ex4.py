import sys
from pathlib import Path
from random import seed
from argon2 import verify_password

import matplotlib.pyplot as plt
from numpy.random import seed as np_seed
from path import Path
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from data_utils import read_digits_data
from knn import KNN
from plot_utils import plot_train_and_test

import numpy as np

random_state = 0
np_seed(9)
seed(9)


def evaluate(knn, X_train, X_test, y_train, y_test, verbose=False, interactive=False):
    if verbose:
        print("Predicting on train data")
    y_predict_train = knn.predict_batch(X_train, show_progress=verbose)
    Ein = accuracy_score(y_train, y_predict_train)

    if verbose:
        print("Predicting on test data")
    y_predict_test = knn.predict_batch(X_test, show_progress=verbose)
    Eout = accuracy_score(y_test, y_predict_test)

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

    Ein, Eout = evaluate(
        knn, X_train, X_test, y_train, y_test, verbose=False, interactive=False
    )

    knn.condense(verbose=False)
    Ein_condensed, Eout_condensed = evaluate(
        knn, X_train, X_test, y_train, y_test, verbose=False, interactive=False
    )

    if verbose:
        print("Full training data...")
        print("Ein:", Ein, "Eout:", Eout)

        print("Condensed data...")
        print("Ein:", Ein_condensed, "Eout:", Eout_condensed)

    return Ein, Eout, Ein_condensed, Eout_condensed


if __name__ == "__main__":
    item = sys.argv[1]

    digits_file = Path("./data/features.train.txt")
    X, y = read_digits_data(digits_file)
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
        Eins, Eouts, Eins_condensed, Eouts_condensed = [], [], [], []
        for _ in tqdm(range(n)):
            Ein, Eout, Ein_condensed, Eout_condensed = experiment(X, y, verbose=True)
            Eins.append(Ein)
            Eouts.append(Eout)
            Eins_condensed.append(Ein_condensed)
            Eouts_condensed.append(Eout_condensed)

        print("AVERAGE RESULTS")

        print("Full training data...")
        print("Ein:", np.mean(Eins), "Eout:", np.mean(Eouts))

        print("Condensed data...")
        print("Ein:", np.mean(Eins_condensed), "Eout:", np.mean(Eouts_condensed))

