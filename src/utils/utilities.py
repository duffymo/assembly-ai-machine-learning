from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

def mse(y_test, predictions):
    return np.mean((y_test-predictions)**2)

def unit_step(x):
    return np.where(x > 0, 1, 0)

def euclidian_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)*(x1-x2)))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def most_common_label(y):
    counter = Counter(y)
    return counter.most_common(1)[0][0]

# see https://urldefense.com/v3/__https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html__;!!NT4GcUJTZV9haA!tQIuUZTLo_aSMuRHQFxSbvf3BQqK4cLCZM5bM6UAignV9iHm2kdUcFG2Dxi3_u1ZTGmSH_Nc9VoA9LUP$ 
# see https://urldefense.com/v3/__https://matplotlib.org/stable/gallery/statistics/histogram_multihist.html__;!!NT4GcUJTZV9haA!tQIuUZTLo_aSMuRHQFxSbvf3BQqK4cLCZM5bM6UAignV9iHm2kdUcFG2Dxi3_u1ZTGmSH_Nc9aqssB0m$ 
def plot_histograms(suptitle, titles, data, nbins=20, plot_width=4.8, plot_height=4.8):
    fig, axes = plt.subplots(nrows=titles.size, ncols=1, figsize=(plot_width, plot_height))
    fig.suptitle(suptitle)
    for i in range(0, titles.size):
            axes[i].set_title(titles[i])
            axes[i].hist(data[titles[i]], bins=nbins)
    fig.tight_layout()
    plt.show()

def plot_learning_curves(model, X, y, test_size=0.2):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    plt.plot(np.sqrt(train_errors), 'r-+', linewidth=2, label='train')
    plt.plot(np.sqrt(val_errors), 'b-', linewidth=3, label='val')