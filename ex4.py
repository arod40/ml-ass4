import sys
from pathlib import Path

import matplotlib.pyplot as plt
from path import Path
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from data_utils import read_digits_data
from knn import KNN
from plot_utils import plot_train_and_test

random_state = 0


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

