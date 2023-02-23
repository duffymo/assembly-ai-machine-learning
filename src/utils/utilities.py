import numpy as np

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

def mse(y_test, predictions):
    return np.mean((y_test-predictions)**2)
