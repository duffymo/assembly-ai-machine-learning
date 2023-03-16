import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import utils.utilities as utils


def visualize_svm(X_data, y_data):

    # see https://pythonprogramming.net/linear-svc-example-scikit-learn-svm-python/
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X_data[:, 0], X_data[:, 1], marker="o", c=y_data)

    x1_min = np.amin(X_data[:, 1])
    x1_max = np.amax(X_data[:, 1])
    ax.set_ylim([x1_min -3, x1_max + 3])

    w = svm.coef_[0]
    xx = np.linspace(-5, 8)
    a = -w[0] / w[1]
    yy = a * xx - svm.intercept_[0] / w[1]
    ax.plot(xx, yy, 'k-', label='non-weighted div')
    plt.show()


if __name__ == '__main__':
    X, y = datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)
    y = np.where(y == 0, -1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    svm = SVC(kernel='linear', C=1.0)
    svm.fit(X_train, y_train)
    predictions = svm.predict(X_test)
    print("Accuracy: ", utils.accuracy(y_test, predictions))
    visualize_svm(X, y)
