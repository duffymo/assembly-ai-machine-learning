import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from sklearn import linear_model

"""
HOML Chapter 4
Closed-form, non-iterative solution for linear regression
"""
class ClosedFormLinearRegression:

    def __init__(self):
        self.coef_ = None

    def fit(self, X, y):
        npoint, ncoef = X.shape
        X_b = np.c_[np.ones((npoint, 1)), X]  # add x0 = 1 to each instance
        self.coef_ = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    def predict(self, X):
        npoint, ncoef = X.shape
        X_new = np.c_[np.ones((npoint, 1)), X]
        return X_new.dot(self.coef_)

class GradientDescent:

    def __init__(self, learning_rate=0.1, max_iterations=1000, tolerance=1.0e-9):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.coef_ = None
        self.num_iterations = 0
        self.eps = 0

    def fit(self, X, y):
        npoint, ncoef = X.shape
        X_b = np.c_[np.ones((npoint, 1)), X]  # add x0 = 1 to each instance
        theta = np.random.randn(1+ncoef, 1)
        for iter in range(self.max_iterations):
            gradients = 2.0 * X_b.T.dot(X_b.dot(theta) - y) / npoint
            self.eps = np.linalg.norm(gradients)
            theta = theta - self.learning_rate * gradients
            my_prediction = X_b.dot(theta)
            plt.plot(X, my_prediction, 'g-')
            if self.eps < self.tolerance:
                self.num_iterations = iter
                break
        self.coef_ = theta

    def predict(self, X):
        npoint, ncoef = X.shape
        X_new = np.c_[np.ones((npoint, 1)), X]
        return X_new.dot(self.coef_)

class StochasticGradientDescent:

    def __init__(self, learning_rate=0.1, num_epochs=50):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.coef_ = None
        self.eta_vs_epoch = np.zeros((num_epochs, 1))

    def fit(self, X, y):
        npoint, ncoef = X.shape
        X_b = np.c_[np.ones((npoint, 1)), X]  # add x0 = 1 to each instance
        theta = np.random.randn(1+ncoef, 1)
        for epoch in range(self.num_epochs):
            for i in range(npoint):
                random_index = np.random.randint(npoint)
                xi = X_b[random_index:random_index+1]
                yi = y[random_index:random_index+1]
                gradients = 2.0 * xi.T.dot(xi.dot(theta) - yi)
                eta = self._learning_schedule(epoch + i)
                self.eta_vs_epoch[epoch] = eta
                theta = theta - eta * gradients
                my_prediction = X_b.dot(theta)
                plt.plot(X, my_prediction, 'g-')
        self.coef_ = theta

    def predict(self, X):
        npoint, ncoef = X.shape
        X_new = np.c_[np.ones((npoint, 1)), X]
        return X_new.dot(self.coef_)

    def _learning_schedule(self, t):
        return self.learning_rate / (1 + self.learning_rate * t)

def generate_noisy_linear_data(xmin, xmax, b, m, npoints):
    """
    Generate linear data with noise
    :param xmin: min value for independent variable xmin <= x <= xmax
    :param xmax: max value for independent variable xmin <= x <= xmax
    :param b: intercept of underlying data with noise
    :param m: slope of underlying data with noise
    :param npoints: number of data points
    :return: (X, y) independent and dependent variable values
    """
    X = xmin + (xmax - xmin) * np.random.rand(npoints, 1)
    y = b + m * X + np.random.randn(npoints, 1)
    return (X, y)


if __name__ == '__main__':
    xmin = 0.0
    xmax = 2.0
    b = 4.0
    m = 3.0
    npoints = 100
    X, y = generate_noisy_linear_data(xmin, xmax, b, m, npoints)

    my_cls = StochasticGradientDescent(num_epochs=1000, learning_rate=0.5)
    my_cls.fit(X, y)
    my_prediction = my_cls.predict(X)
    print('My coeffs    : ', my_cls.coef_)
    print('# epochs     : ', my_cls.num_epochs)
    print('learning rate: ', my_cls.learning_rate)

    eta_min = np.min(my_cls.eta_vs_epoch)
    eta_max = np.max(my_cls.eta_vs_epoch)
    plt.plot(range(my_cls.num_epochs), my_cls.eta_vs_epoch, 'b-')
    plt.axis([0, my_cls.num_epochs, eta_min, eta_max])
    plt.show()

    sk_cls = linear_model.LinearRegression()
    sk_cls.fit(X, y)
    sk_prediction = sk_cls.predict(X)
    print('SK coeffs: ', sk_cls.intercept_, sk_cls.coef_)

    intercept_diff = my_cls.coef_[0] - sk_cls.intercept_
    slope_diff = my_cls.coef_[1] - sk_cls.coef_[0]
    print('Underlying intercept: ', b)
    print('Stochastic intercept: ', my_cls.coef_[0])
    print('SK linreg  intercept: ', sk_cls.intercept_)
    print('Intercept difference: ', intercept_diff)
    print('Intercept % error   : ', 100.0 * intercept_diff / sk_cls.intercept_, '%')
    print('Underlying slope    : ', m)
    print('Stochastic slope    : ', my_cls.coef_[1])
    print('SK linreg slope     : ', sk_cls.coef_[0])
    print('Slope difference    : ', slope_diff)
    print('Slope % error       : ', 100.0 * slope_diff / sk_cls.coef_[0], '%')

    ymin = np.min(y)
    ymax = np.max(y)
    plt.plot(X, y, 'b.')
    plt.plot(X, my_prediction, 'r-')
    plt.plot(X, sk_prediction, 'g--')
    plt.axis([0, xmax, ymin, ymax])
    plt.show()

