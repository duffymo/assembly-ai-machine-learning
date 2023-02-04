
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split


def euclidian_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # compute the distance
        distances = [euclidian_distance(x, x_train) for x_train in self.X_train]
        # get the closest k
        k_indicies = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indicies]
        # majority vote
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]

def train():
    cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    plt.figure()
    plt.scatter(X[:, 2], X[:, 3], c=y, cmap=cmap, edgecolor='k', s=20)
    plt.show()

    classifier = KNN(k=5)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    print(predictions)

    accuracy = np.sum(predictions == y_test) / len(y_test)
    print(accuracy)

if __name__ == '__main__':
    train()


