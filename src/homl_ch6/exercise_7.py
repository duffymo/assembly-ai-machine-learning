from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':
    X, y = datasets.make_moons(n_samples=10_000, noise=0.4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    tree_clf = DecisionTreeClassifier()
    param_grid = [
        {
            'max_depth': [2, 3, 4, 5, 6],
            'max_leaf_nodes': [5, 10, 15, 20],
            'min_weight_fraction_leaf': [0.0, 0.1, 0.2, 0.3]
        }
    ]
    grid_search = GridSearchCV(tree_clf, param_grid)
    grid_search.fit(X, y)
    print('Decision tree grid search')
    print('Best score    : ', grid_search.best_score_)
    print('Best params   : ', grid_search.best_params_)
    print('Best estimator: ', grid_search.best_estimator_)
    print('Best index    : ', grid_search.best_index_)
