import matplotlib.pyplot as plt
import numpy as np

class PCA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.component = None
        self.mean = None
        self.lambdas = None

    def fit(self, X):
        # mean centering
        self.mean = np.mean(X, axis=0)
        # Make sure n_components can't be larger than number of features.
        self.n_components = np.min([self.n_components, X.shape[1]])
        X = X - self.mean
        cov = np.cov(X.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        # eigenvectors are column vectors; transpose to match features
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        self.components = eigenvectors[:self.n_components]
        self.lambdas = eigenvalues[:self.n_components]

    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components.T)

if __name__ == "__main__":
    from sklearn import datasets
    data = datasets.load_iris()
    X = data.data
    y = data.target

    pca = PCA(2)
    pca.fit(X)
    X_projected = pca.transform(X)
    print("Shape of X: ", X.shape)
    print("Shape of transformed X: ", X_projected.shape)
    print("Shape of eigenvalues: ", pca.lambdas.shape)
    print(pca.lambdas)
    print(pca.components.T)

    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    plt.scatter(x1, x2, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3))
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar()
    plt.show()
