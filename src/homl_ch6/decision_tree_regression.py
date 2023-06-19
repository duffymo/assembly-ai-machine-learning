import matplotlib.pyplot as plt
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz

def noisy_quadratic(xmin, xmax, c0, c1, c2, amplitude, npoints):
    dx = (xmax - xmin) / npoints
    X = np.arange(xmin, xmax, dx)
    signal = [c0 + x * (c1 + x*c2) for x in X]
    noise = amplitude * np.random.normal(size=X.size)
    return X, signal, noise

if __name__ == '__main__':
    X, signal, noise = noisy_quadratic(0.0, 1.0, 1.0, -4.0, 4.0, 0.1, 100)
    y = signal + noise

    tree_reg = DecisionTreeRegressor(max_depth=3)
    tree_reg.fit(X.reshape(-1, 1), y)

    y_predict = tree_reg.predict(X.reshape(-1, 1))

    export_graphviz(tree_reg,
                    out_file='noisy_quadratic_tree.dot',
                    rounded=True,
                    filled=True,
                    impurity=True)

    plt.plot(X, signal, 'r')
    plt.plot(X, y, 'bo')
    plt.plot(X, y_predict, 'g')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()

