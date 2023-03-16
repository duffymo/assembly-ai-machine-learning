# see https://scikit-learn.org/stable/modules/preprocessing.html

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    pipeline = make_pipeline(StandardScaler(), LogisticRegression())
    pipeline.fit(X_train, y_train)
    acc = pipeline.score(X_test, y_test)
    print(acc)


