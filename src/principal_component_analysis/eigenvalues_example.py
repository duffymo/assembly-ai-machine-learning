import numpy as np

# see https://scriptverse.academy/tutorials/python-eigenvalues-eigenvectors.html

if __name__ == "__main__":
    a = np.array([[3, 1], [2, 2]])
    w, v = np.linalg.eig(a)

    print(w) # eigenvalues
    print(v) # normalized eigenvectors

    # check the identity for the first eigenvalue
    lhs = np.dot(a, v[:, 0])
    rhs = np.dot(w[0], v[:, 0])
    print("1st eigenvalue check: ", np.allclose(lhs, rhs))
    print("1st eigenvector norm check: ", np.linalg.norm(v[:, 0]))

    # check the identity for the second eigenvalue
    lhs = np.dot(a, v[:, 1])
    rhs = np.dot(w[1], v[:, 1])
    print("2nd eigenvalue check: ", np.allclose(lhs, rhs))
    print("2nd eigenvector norm check: ", np.linalg.norm(v[:, 1]))


