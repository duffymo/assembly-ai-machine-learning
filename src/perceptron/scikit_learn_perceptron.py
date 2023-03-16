import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split

import utils.utilities as utils

if __name__ == '__main__':
    X, y = datasets.make_blobs(n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    p = Perceptron(alpha=0.0001, max_iter=1000)
    p.fit(X_train, y_train)
    predictions = p.predict(X_test)
    print("Accuracy: ", utils.accuracy(y_test, predictions))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)

    x0_1 = np.amin(X_train[:, 0])
    x0_2 = np.amax(X_train[:, 0])
    x1_1 = (-p.coef_[:, 0] * x0_1 - p.intercept_[0]) / p.coef_[:, 1]
    x1_2 = (-p.coef_[:, 0] * x0_2 - p.intercept_[0]) / p.coef_[:, 1]

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

    ymin = np.amin(X_train[:, 1])
    ymax = np.amax(X_train[:, 1])
    ax.set_ylim([-15, 5])

    plt.show()


