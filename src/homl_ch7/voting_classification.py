from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

if __name__ == '__main__':

    num_samples = 10_000
    rs = 1945
    X, y = datasets.make_moons(n_samples=num_samples, noise=0.4, random_state=rs)
    overall_train_size = 8_000
    overall_test_size = num_samples - overall_train_size
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=overall_train_size, random_state=rs)

    log_clf = LogisticRegression()
    rf_clf = RandomForestClassifier()
    svm_clf = SVC()
    svm_clf.probability = True
    voting_clf = VotingClassifier(
        estimators=[('lr', log_clf), ('rf', rf_clf), ('svm', svm_clf)],
        voting='soft'
    )

    for clf in (log_clf, rf_clf, svm_clf, voting_clf):
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        print(clf.__class__.__name__, accuracy_score(y_test, y_predict))

