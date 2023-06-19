import numpy as np
import scipy.stats as stats
from sklearn import datasets
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.tree import DecisionTreeClassifier

from utils.utilities import accuracy

if __name__ == '__main__':
    num_samples = 10_000
    rs = 1945
    X, y = datasets.make_moons(n_samples=num_samples, noise=0.4, random_state=rs)
    overall_train_size = 8_000
    overall_test_size = num_samples - overall_train_size
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=overall_train_size, random_state=rs)

    omd = 6
    omln = 20
    overall_tree_classifier = DecisionTreeClassifier(max_depth=omd, max_leaf_nodes=omln)
    overall_tree_classifier.fit(X_train, y_train)
    overall_y_predict = overall_tree_classifier.predict(X_test)
    overall_accuracy = accuracy(y_test, overall_y_predict)

    # create subtrees
    num_subtrees = 1_000
    ss = ShuffleSplit(n_splits=num_subtrees, train_size=100, random_state=rs)
    sub_tree_classifiers = [DecisionTreeClassifier(max_depth=omd, max_leaf_nodes=omln).fit(X[sub_train], y[sub_train]) for sub_train, sub_test in ss.split(X_train, y_train)]
    sub_y_predict = []
    sub_accuracies = []
    for k, clf in enumerate(sub_tree_classifiers):
        sub_y_predict.append(clf.predict(X_test))
        sub_accuracies.append(accuracy(y_test, sub_y_predict[k]))
    sub_accuracies.sort(reverse=True)
    superstars = (sub_accuracies > overall_accuracy).sum()
    laggards = (sub_accuracies < overall_accuracy).sum()

    # random forest
    voted_predictions = stats.mode(sub_y_predict, axis=0, keepdims=False)

    print('Overall accuracy: ', overall_accuracy)
    print('Max sub accuracy: ', np.max(sub_accuracies))
    print('Superstars      : ', superstars)
    print('Sub-performers  : ', laggards)
    print('Voted accuracy  : ', accuracy(y_test, voted_predictions))

