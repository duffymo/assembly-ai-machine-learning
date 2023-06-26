import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


def noisy_quadratic(xmin, xmax, c0, c1, c2, amplitude, npoints):
    dx = (xmax - xmin) / npoints
    X = np.arange(xmin, xmax, dx)
    signal = [c0 + x * (c1 + x * (c2 * x)) for x in X]
    noise = amplitude * np.random.normal(size=X.size)
    return X.reshape(npoints, 1), signal, noise


if __name__ == '__main__':
    rs = 1945
    num_samples = 1_000
    overall_train_size = 800
    overall_test_size = num_samples - overall_train_size
    X, signal, noise = noisy_quadratic(0.0, 1.0, 1.0, -4.0, 4.0, 0.1, num_samples)
    y = signal + noise
    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, train_size=overall_train_size, random_state=rs)

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_predict = lin_reg.predict(X_test)
    print('intercept   : ', lin_reg.intercept_)
    print('coefficients: ', lin_reg.coef_)

    plt.figure()
    plt.plot(X_train[:, 1], y_train, 'b^', label='train')
    plt.plot(X_test[:, 1], y_predict, 'gd', label='test')
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend(loc='best')
    plt.title('Regression predictions')
    plt.show()
