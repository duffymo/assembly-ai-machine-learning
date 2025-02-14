import collections

import numpy as np
from keras.datasets import mnist

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    train_counts = collections.Counter(y_train)
    test_counts = collections.Counter(y_test)
    total_counts = collections.Counter(np.concatenate((y_train, y_test)))
    print("training set size: {}".format(X_train.shape[0]))
    print("test set size    : {}".format(X_test.shape[0]))
    print("train counts     : {}".format(sorted(train_counts.items())))
    print("test counts      : {}".format(sorted(test_counts.items())))
    print("total counts     : {}".format(sorted(total_counts.items())))
