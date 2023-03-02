import numpy as np

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

def mse(y_test, predictions):
    return np.mean((y_test-predictions)**2)
def euclidian_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)*(x1-x2)))

