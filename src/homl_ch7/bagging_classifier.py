import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier



if __name__ == '__main__':
    X, y = datasets.make_moons(n_samples=10_000, noise=0.4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    bag_clf = BaggingClassifier(DecisionTreeClassifier(max_depth=3),
                                n_estimators=500,
                                max_samples=100,
                                bootstrap=True,
                                n_jobs=1,
                                oob_score=True)
    bag_clf.fit(X_train, y_train)
    print('oob score: ', bag_clf.oob_score_)
    y_predict = bag_clf.predict(X_test)
    print('accuracy : ', accuracy_score(y_test, y_predict))

    # How to draw decision boundary
    # https://stackoverflow.com/questions/43778380/how-to-draw-decision-boundary-in-svm-sklearn-data-in-python
    npoints = 1_000
    xmin, xmax = X[:, 0].min() - 1, X[:, 0].max() + 1
    ymin, ymax = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(xmin, xmax, (xmax - xmin) / npoints),
        np.arange(ymin, ymax, (ymax - ymin) / npoints)
    )
    Z = bag_clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.show()
