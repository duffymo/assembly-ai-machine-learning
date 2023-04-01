# HOML chapter 3: MNIST digit classification without neural networks.
# Downloaded data by hand from https://urldefense.com/v3/__https://www.kaggle.com/datasets/vikramtiwari/mnist-numpy__;!!NT4GcUJTZV9haA!pQAFY8s7jhZ2xMRhUoyvFt9b-bFnC3Zkp0pn0wHf_KoZmO6weE3GN3ccA1EEGXk3vzftYIya-iWf6i4a$ 
# Somehow I managed to get Tensorflow installed after several failures.  I don't know why.

import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

from utils.utilities import accuracy

"""
Multi-class classifier using stochastic gradient descent
HOML chapter 3
"""
if __name__ == '__main__':

    start = time.time()

    with np.load('../../resources/mnist.npz') as f:
        X_train, y_train, X_test, y_test = f['x_train'], f['y_train'], f['x_test'], f['y_test']
    print('X_train shape: ', X_train.shape)
    print('y_train shape: ', y_train.shape)
    print('X_test  shape: ', X_test.shape)
    print('y_test  shape: ', y_test.shape)

    shuffle_index = np.random.permutation(X_train.shape[0])
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

    train_dist = np.unique(y_train, return_counts=True)
    print('Train distribution: ', train_dist)
    test_dist = np.unique(y_test, return_counts=True)
    print('Test  distribution: ', test_dist)

    ntrain_samples, nx, ny = X_train.shape
    X_train = X_train.reshape(ntrain_samples, nx * ny)
    ntest_samples, nx, ny = X_test.shape
    X_test = X_test.reshape(ntest_samples, nx * ny)
    finish = time.time()
    print('Data prep time (sec): ', (finish-start))
    start = finish

    rs=1922
#    classifier = SGDClassifier(random_state=rs)
    classifier = RandomForestClassifier(max_depth=100, random_state=rs)
    classifier.fit(X_train, y_train)
    finish = time.time()
    print('Model training time (sec): ', (finish-start))
    start = finish

    y_train_pred = cross_val_predict(classifier, X_train, y_train, cv=3)
    finish = time.time()
    print('Cross validation prediction time (sec): ', (finish-start))
    start = finish

    confusion = confusion_matrix(y_train, y_train_pred)
    confusion_row_sums = confusion.sum(axis=1, keepdims=True)
    normalized_confusion_matrix = confusion / confusion_row_sums
    np.fill_diagonal(normalized_confusion_matrix, 0)
    plt.matshow(normalized_confusion_matrix, cmap=plt.colormaps['Spectral_r'])
    plt.show()
    finish = time.time()
    print('Confusion matrix time (sec): ', (finish-start))
    start = finish

    predictions = classifier.predict(X_test)
    acc = accuracy(y_test, predictions)
    print('Accuracy: ', acc)
    finish = time.time()
    print('Total elapsed time (sec): ', (finish-start))
