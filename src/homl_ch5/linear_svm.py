import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import LinearSVC


def classify_iris():
    iris = datasets.load_iris()
    X = iris['data'][:, (2, 3)] # petal length & width
    y = (iris['target'] == 2).astype(np.float64) # Iris-Virginica

    # noinspection PyTypeChecker
    svm_clf = Pipeline((
        ('scalar', StandardScaler()),
        ('linear_svc', LinearSVC(C=1, loss='hinge'))
    ))

    svm_clf.fit(X, y)
    y_prediction = svm_clf.predict(X)

    plt.scatter(X, y, 'g.')
    plt.scatter(X, y_prediction, 'b.')
    plt.show()

def classify_moons():
    # https://medium.com/mlearning-ai/how-to-create-a-two-moon-dataset-and-make-predictions-on-it-dcc090c829af
    moons = datasets.make_moons(n_samples=100, noise=0.1)
    X, y = moons[0], moons[1]
    df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
    colors = { 0: 'red', 1: 'blue' }
    fig, ax = plt.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])

    polynomial_svm_clf = Pipeline((
        ('poly_features', PolynomialFeatures(degree=3)),
        ('scalar', StandardScaler()),
        ('svm_clf', LinearSVC(C=10, loss='hinge'))
    ))
    polynomial_svm_clf.fit(X, y)

    # see https://stackoverflow.com/questions/43778380/how-to-draw-decision-boundary-in-svm-sklearn-data-in-python
    npoints = 1000
    xmin, xmax = X[:, 0].min() - 1, X[:, 0].max() + 1
    ymin, ymax = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(xmin, xmax, (xmax - xmin) / npoints),
        np.arange(ymin, ymax, (ymax - ymin) / npoints)
    )
    Z = polynomial_svm_clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.show()


if __name__ == '__main__':
    classify_moons()
