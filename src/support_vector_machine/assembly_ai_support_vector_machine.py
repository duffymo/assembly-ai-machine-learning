import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

import utils.utilities as utils


class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y < 0, -1, 1)
        self.w = np.random.rand(n_features)
        self.b = np.random.rand()
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * np.dot(x_i, self.w - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2.0 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (2.0 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

def get_hyperplane_value(x, w, b, offset):
    return (-w[0] * x + b + offset) / w[1]

def visualize_svm(X_data, y_data):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X_data[:, 0], X_data[:, 1], marker="o", c=y_data)

    x0_1 = np.amin(X_data[:, 0])
    x0_2 = np.amax(X_data[:, 0])

    x1_1 = get_hyperplane_value(x0_1, svm.w, svm.b, 0)
    x1_2 = get_hyperplane_value(x0_2, svm.w, svm.b, 0)

    x1_1_m = get_hyperplane_value(x0_1, svm.w, svm.b, -1)
    x1_2_m = get_hyperplane_value(x0_2, svm.w, svm.b, -1)

    x1_1_p = get_hyperplane_value(x0_1, svm.w, svm.b, 1)
    x1_2_p = get_hyperplane_value(x0_2, svm.w, svm.b, 1)

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

    x1_min = np.amin(X_data[:, 1])
    x1_max = np.amax(X_data[:, 1])
    ax.set_ylim([x1_min -3, x1_max + 3])

    plt.show()


if __name__ == '__main__':
    X, y = datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)
    y = np.where(y == 0, -1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    svm = SVM()
    svm.fit(X_train, y_train)
    predictions = svm.predict(X_test)
    print("Accuracy: ", utils.accuracy(y_test, predictions))
    visualize_svm(X, y)
