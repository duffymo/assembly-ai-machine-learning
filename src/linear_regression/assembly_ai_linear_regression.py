import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from utils.utilities import mse

"""
Calculate best linear fit for a resources set using gradient descent.
Learning rate determines how quickly weights and bias are updated.
Could you use auto gradient ideas to figure out how to determine optimal learning rate?
"""
class LinearRegression:

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
            y_pred = np.dot(X, self.weights) + self.bias
            dw = np.dot(X.T, (y_pred-y)) / n_samples
            db = np.sum(y_pred-y) / n_samples
            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
"""
Calculate optimal learning rate.
Create a dictionary of learning rates and MSE to optimize.
"""
def optimize_learning_rate(X_train, X_test, y_train, y_test, lr_min, lr_max, n_rates):
    optimize = {}
    rates = np.linspace(lr_min, lr_max, num=n_rates)
    for rate in rates:
        reg = LinearRegression(rate)
        reg.fit(X_train, y_train)
        predictions = reg.predict(X_test)
        err = mse(y_test, predictions)
        optimize[rate] = err
    return optimize


def train():
    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    # Determine the optimal learning rate.
    optimize = optimize_learning_rate(X_train, X_test, y_train, y_test, lr_min=0.0001, lr_max=0.01, n_rates=30)
    print(optimize)
    min_error = min(optimize.values())
    best_learning_rate = [key for key in optimize if optimize[key] == min_error]
    print('Best learning rate: ', best_learning_rate)
    print('Minimum error     : ', min_error)

    plt.plot(optimize.keys(), optimize.values(), color='red', linewidth=2, label='MSE versus LR')
    plt.show()

    # Use the optimal learning rate
    reg = LinearRegression(lr=0.00453793103448276)
    reg.fit(X_train, y_train)
    predictions = reg.predict(X_test)
    err = mse(y_test, predictions)
    print("Error: ", err)
    print("Bias: ", reg.bias)
    print("Weights: ", reg.weights)

    # Plot the best regression line
    y_pred_line = reg.predict(X)
    cmap = plt.get_cmap('viridis')
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
    plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction')
    plt.show()

if __name__ == '__main__':
    train()
