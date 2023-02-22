from collections import Counter

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from decision_trees.assembly_ai_decision_trees import DecisionTree


class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                n_features=self.n_features)
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        return np.array([self._most_common_label(pred) for pred in tree_preds])

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

def train():
    ds = datasets.load_breast_cancer()
    X, y = ds.data, ds.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    clf = RandomForest(n_trees=20)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    acc = accuracy(y_test, predictions)
    print(acc)


if __name__ == '__main__':
    train()
