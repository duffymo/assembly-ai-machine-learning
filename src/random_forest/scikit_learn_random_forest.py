from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import utils.utilities as utils

if __name__ == '__main__':
    ds = datasets.load_breast_cancer()
    X, y = ds.data, ds.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    clf = RandomForestClassifier(max_depth=100, random_state=0)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    acc = utils.accuracy(y_test, predictions)
    print(acc)
