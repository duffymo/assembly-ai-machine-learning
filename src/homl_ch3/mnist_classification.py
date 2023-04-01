# HOML chapter 3: MNIST digit classification without neural networks.
# Downloaded data by hand from https://urldefense.com/v3/__https://www.kaggle.com/datasets/vikramtiwari/mnist-numpy__;!!NT4GcUJTZV9haA!pQAFY8s7jhZ2xMRhUoyvFt9b-bFnC3Zkp0pn0wHf_KoZmO6weE3GN3ccA1EEGXk3vzftYIya-iWf6i4a$ 
# Somehow I managed to get Tensorflow installed after several failures.  I don't know why.

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, confusion_matrix, precision_score, \
    recall_score, f1_score
from sklearn.model_selection import cross_val_predict

from utils.utilities import accuracy


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
    plt.xlabel('Threshold')
    plt.legend(loc='upper left')
    plt.ylim([0, 1])

def binary_classifier(X_train, y_train, X_test, y_test, digit=5, rs=1922):

    # Binary classifier to choose the digit
    y_train_digit = (y_train == digit)
    y_test_digit = (y_test == digit)

    sgd = SGDClassifier(random_state=rs)
    ntrain_samples, nx, ny = X_train.shape
    X_train_digit = X_train.reshape(ntrain_samples, nx * ny)
    ntest_samples, nx, ny = X_test.shape
    X_test_digit = X_test.reshape(ntest_samples, nx * ny)

#    calc_accuracy(sgd, X_train_digit, y_train_digit, X_test_digit, y_test, digit)
#    plot_precision_vs_recall(sgd, X_train_digit, y_train_digit, digit)
    plot_roc(sgd, X_train_digit, y_train_digit, str(digit))

def calc_accuracy(sgd, X_train, y_train, X_test, y_test, digit='all'):
    sgd.fit(X_train, y_train)
    ntest_samples, nx, ny = X_test.shape
    X_test_digit = X_test.reshape(ntest_samples, nx*ny)
    predictions = sgd.predict(X_test_digit)
    acc = accuracy(y_test, predictions)
    print('Digit {0} accuracy: {1}'.format(digit, acc))
    y_train_pred = cross_val_predict(sgd, X_train, y_train, cv=3)
    confusion = confusion_matrix(y_train, y_train_pred)
    print('Confusion matrix              : ', confusion)
    print('Precision score  for digit {0}: {1}'.format(digit, precision_score(y_train, y_train_pred)))
    print('Recall    score  for digit {0}: {1}'.format(digit, recall_score(y_train, y_train_pred)))
    print('F1        score  for digit {0}: {1}'.format(digit, f1_score(y_train, y_train_pred)))

def plot_precision_vs_recall(sgd, X_train, y_train, digit='all'):
    print('Precision vs recall plot for digit {}'.format(digit))
    y_scores = cross_val_predict(sgd, X_train, y_train, cv=3, method='decision_function')
    precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)
    plt.plot(recalls, precisions, 'g-')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()

def plot_roc(sgd, X_train, y_train, digit='all'):
    y_scores = cross_val_predict(sgd, X_train, y_train, cv=3, method='decision_function')
    fpr, tpr, thresholds = roc_curve(y_train, y_scores)
    ra_score = roc_auc_score(y_train, y_scores)
    l = 'ROC plot for digit {0}: {1}'.format(digit, ra_score)
    print(l)
    plt.plot(fpr, tpr, linewidth=2, label=l)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()



def show_digit(digit):
    digit_image = digit.reshape(28, 28)
    plt.imshow(digit_image, cmap=matplotlib.colormaps.binary, interpolation='nearest')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':

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

    # idx = 36_000
    # show_digit(X_train[idx])
    # print(y_train[idx])

    for i in range(0, 10):
        binary_classifier(X_train, y_train, X_test, y_test, i)


