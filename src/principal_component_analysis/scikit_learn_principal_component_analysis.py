import matplotlib.pyplot as plt
# unused but required import for doing 3d projections with matplotlib < 3.2
import mpl_toolkits.mplot3d  # noqa
from sklearn import decomposition

if __name__ == "__main__":
    from sklearn import datasets
    data = datasets.load_iris()
    X = data.data
    y = data.target

    pca = decomposition.PCA(n_components=2)
    pca.fit(X)
    X_projected = pca.transform(X)
    print("Shape of X: ", X.shape)
    print("Shape of transformed X: ", X_projected.shape)

    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    plt.scatter(x1, x2, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3))
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar()
    plt.show()
