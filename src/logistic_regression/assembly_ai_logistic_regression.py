import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from utils.utilities import sigmoid, accuracy


# logistic regression
# sigmoid
# cross entropy
class LogisticRegression:

    def __init__(self, lr=0.001, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            linear_pred = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(linear_pred)
            dw = np.dot(X.T, (y_pred-y)) / n_samples
            db = np.sum(y_pred-y) / n_samples
            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db

    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y <= 0.5 else 1 for y in y_pred]
        return class_pred


"""
Calculate optimal learning rate.
Create a dictionary of learning rates and MSE to optimize.
"""
def optimize_learning_rate(X_train, X_test, y_train, y_test, lr_min, lr_max, n_rates):
    optimize = {}
    rates = np.linspace(lr_min, lr_max, num=n_rates)
    for rate in rates:
        reg = LogisticRegression(rate)
        reg.fit(X_train, y_train)
        predictions = reg.predict(X_test)
        acc = accuracy(y_test, predictions)
        optimize[rate] = acc
    return optimize


def train():
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


    # Determine the optimal learning rate.
    optimize = optimize_learning_rate(X_train, X_test, y_train, y_test, lr_min=0.005, lr_max=0.009, n_rates=20)
    print(optimize)
    max_accuracy = max(optimize.values())
    best_learning_rate = [key for key in optimize if optimize[key] == max_accuracy]
    print('Best learning rate: ', best_learning_rate)
    print('Maximum accuracy  : ', max_accuracy)

    plt.plot(optimize.keys(), optimize.values(), color='red', linewidth=2, label='MSE versus LR')
    plt.show()

    # Use the optimal learning rate.
    max_best_learning_rate = np.max(best_learning_rate)
    print('Max best learning rate: ', max_best_learning_rate)
    clf = LogisticRegression(lr=max_best_learning_rate)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    acc = accuracy(y_test, predictions)
    print(acc)


if __name__ == '__main__':
    train()
