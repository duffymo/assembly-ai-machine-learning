import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import utils.utilities as utils

if __name__ == '__main__':
    ds = datasets.load_breast_cancer()
    X, y = ds.data, ds.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    clf = DecisionTreeClassifier(max_depth=100)
    clf = clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    acc = utils.accuracy(y_test, predictions)
    print("Accuracy: ", acc)
    # see https://mljar.com/blog/visualize-decision-tree/
    fig = plt.figure(figsize=(50, 25))
    _ = tree.plot_tree(clf, filled=True)
    plt.show()

