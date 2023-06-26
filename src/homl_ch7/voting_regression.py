import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR



def noisy_quadratic(xmin, xmax, c0, c1, c2, amplitude, npoints):
    dx = (xmax - xmin) / npoints
    X = np.arange(xmin, xmax, dx)
    signal = [c0 + x * (c1 + x * (c2 * x)) for x in X]
    noise = amplitude * np.random.normal(size=X.size)
    return X, signal, noise




if __name__ == '__main__':

    rs = 1945
    num_samples = 1_000
    overall_train_size = 800
    overall_test_size = num_samples - overall_train_size
    X, signal, noise = noisy_quadratic(0.0, 1.0, 1.0, -4.0, 4.0, 0.1, num_samples)
    y = signal + noise
    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(X.reshape(num_samples, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=overall_train_size, random_state=rs)
    X_poly_train, X_poly_test, y_poly_train, y_poly_test = train_test_split(X_poly, y, train_size=overall_train_size, random_state=rs)

    lin_reg = LinearRegression()
    rf_reg = RandomForestRegressor()
    svm_reg = SVR()
    voting_reg = VotingRegressor(
        estimators=[('lr', lin_reg), ('rf', rf_reg), ('svm', svm_reg)],
    )
    y_predict = []
    lin_reg.fit(X_poly_train, y_poly_train)
    y_predict.append(lin_reg.predict(X_poly_test))
    for reg in (rf_reg, svm_reg, voting_reg):
        reg.fit(X_train.reshape(-1, 1), y_train)
        y_predict.append(reg.predict(X_test.reshape(-1, 1)))

    plt.figure()
    formats = ['gd', 'b^', 'ys', 'r*']
    labels = ['lin', 'rf', 'svm', 'voting']
    for i, format in enumerate(formats):
        plt.plot(X_test, y_predict[i], format, label=labels[i])
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.xlabel('Training samples')
    plt.ylabel('Predicted')
    plt.legend(loc='best')
    plt.title('Regressor Predictions')
    plt.show()
