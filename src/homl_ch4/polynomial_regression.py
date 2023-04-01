import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

from utils.utilities import plot_learning_curves


def generate_noisy_quadratic_data(xmin, xmax, a, b, c, npoints):
    """
    Generate quadratic data with noise
    :param xmin: min value for independent variable xmin <= x <= xmax
    :param xmax: max value for independent variable xmin <= x <= xmax
    :param a: coefficient for x^2 term
    :param b: coefficient for x term
    :param c: coefficient for constant term
    :param npoints: number of points to generate
    :return: (X, y) independent and dependent variable values
    """
    X = xmin + (xmax-xmin) * np.random.rand(npoints, 1)
    y = a * X**2 + b * X + c + np.random.randn(npoints, 1)
    return (X, y)



if __name__ == '__main__':
    xmin = -3.0
    xmax = 3.0
    a = 0.5
    b = 1.0
    c = 2.0
    npoints = 100
    X, y = generate_noisy_quadratic_data(xmin, xmax, a, b, c, npoints)
    poly_features = PolynomialFeatures(degree=10, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    sk_lin_reg = linear_model.LinearRegression()
    sk_lin_reg.fit(X_poly, y)
    sk_prediction = sk_lin_reg.predict(X_poly)

    print('Underlying x^2   coeff: ', a)
    print('Calculated x^2   coeff: ', sk_lin_reg.coef_[:,1])
    print('Underlying x     coeff: ', b)
    print('Calculated x     coeff: ', sk_lin_reg.coef_[:,0])
    print('Underlying const coeff: ', c)
    print('Calculated const coeff: ', sk_lin_reg.intercept_)

    ymin = np.min(y)
    ymax = np.max(y)
    plt.plot(X, y, 'b.')
    plt.plot(X, sk_prediction, 'g.')
    plt.axis([xmin, xmax, ymin, ymax])
    plt.show()

    plot_learning_curves(sk_lin_reg, X_poly, y)
    plt.axis([0, 80, 0, 3])
    plt.show()