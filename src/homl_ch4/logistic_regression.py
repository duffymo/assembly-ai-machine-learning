import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import LogisticRegression

"""
HOML Chapter 4: Logistic Regression
"""
if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris['data'][:, (2, 3)]  # petal length & width
    y = iris['target']

    softmax_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=10)
    softmax_reg.fit(X, y)

    y_probability = softmax_reg.predict_proba(X)
    plt.scatter(X[:, 0], X[:, 1], cmap=plt.colormaps['Spectral_r'])
    plt.show()

    plt.plot(X, y_probability[:, 0], 'g.')
    plt.plot(X, y_probability[:, 1], 'b.')
    plt.plot(X, y_probability[:, 2], 'r.')
    plt.show()


