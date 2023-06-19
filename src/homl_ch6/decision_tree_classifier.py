from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

if __name__ == '__main__':
    iris = load_iris()
    X = iris.data[:, 2:] # petal length and width
    y = iris.target

    tree_clf = DecisionTreeClassifier(max_depth=3)
    tree_clf.fit(X, y)

    export_graphviz(tree_clf,
                    out_file='iris_tree.dot',
                    feature_names=iris.feature_names[2:],
                    rounded=True,
                    filled=True,
                    impurity=True)
