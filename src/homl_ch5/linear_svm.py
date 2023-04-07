import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
    plt.show()
    return X, y

if __name__ == '__main__':
    X, y = classify_moons()
