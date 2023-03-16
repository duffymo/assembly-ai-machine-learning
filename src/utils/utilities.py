from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


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

# see https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
# see https://matplotlib.org/stable/gallery/statistics/histogram_multihist.html
def plot_histograms(suptitle, titles, data, nbins=20, plot_width=4.8, plot_height=4.8):
    fig, axes = plt.subplots(nrows=titles.size, ncols=1, figsize=(plot_width, plot_height))
    fig.suptitle(suptitle)
    for i in range(0, titles.size):
            axes[i].set_title(titles[i])
            axes[i].hist(data[titles[i]], bins=nbins)
    fig.tight_layout()
    plt.show()
