import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


def plot(clusters, X, centroids):
    fix, ax = plt.subplots(figsize=(12, 8))
    for i, index in enumerate(clusters):
        point = X[index].T
        ax.scatter(*point)

    for point in centroids:
        ax.scatter(*point, marker='x', color='black', linewidth=2)
    plt.show()

if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    X, y = make_blobs(centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40)
    print('X shape: ', X.shape)

    clusters = len(np.unique(y))
    print('Clusters: ', clusters)

    km = KMeans(n_clusters=clusters)
    y_pred = km.fit(X, y)
    print(km.cluster_centers_)
#    plot(clusters, X, km.cluster_centers_)

