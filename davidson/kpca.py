import numpy as np
from nptyping import Float64, NDArray
from sklearn.datasets import make_moons
from davidson import Davidson
import matplotlib.pyplot as plt
import time

def gaussian_rbf_kernel(xi, xj, gamma):
    """Gaussian radial basis function kernel"""
    displacement = xi - xj
    euclidean_distance = np.linalg.norm(displacement)
    sqdist = euclidean_distance**2
    k_ij = np.exp(-1*gamma*sqdist)
    return k_ij

def compute_kernel_matrix(data, gamma=15.0):
    """Return the kernel matrix"""
    n_points = len(data)
    K = np.zeros((n_points, n_points))
    # populate the kernel matrix
    # this is a covariance matrix in the high-dimensional
    # space specified by the kernel
    for i in range(n_points):
        for j in range(i,n_points):
            k_ij = gaussian_rbf_kernel(data[i], data[j], gamma)
            K[i,j] = k_ij
            K[j,i] = k_ij
    return K

def center_kernel_matrix(K):
    """center the kernel matrix"""
    n = K.shape[0]
    ones = np.ones((n,n)) / n
    K_centered = K - ones.dot(K) - K.dot(ones) + ones.dot(K).dot(ones)
    return K_centered

def compute_eigendecomposition(K, n_roots):
    # note that the 1st eigenvector is trivial
    # reorder evals and evecs
    evals, evecs = np.linalg.eigh(K)
    evals = [evals[-i] for i in range(1, n_roots+1)]
    evecs = np.column_stack([evecs[:,-i] for i in range(1, n_roots+1)])
    return evals, evecs


if __name__=='__main__':
    plt.figure(figsize=(6,5))
    X, y = make_moons(n_samples=5000, random_state=0)
    start = time.time()
    K = compute_kernel_matrix(X)
    K_centered = center_kernel_matrix(K)
    end = time.time()
    print(f'Time to build kernel matrix: {end-start:<.8f}s elapsed')

    start_np = time.time()
    evals_np, evecs_np = compute_eigendecomposition(K_centered, 2)
    end_np = time.time()
    print(f'Kernel PCA by NumPy: {end_np-start_np:<.8f}s elapsed')
    
    start = time.time()
    davidson = Davidson()
    evals_dav, evecs_dav = davidson.find_roots(
            K_centered, 
            n_roots=5, 
            largest_roots=True,
            precondition=False,
            verbose=False
            )
    end = time.time()
    print(f'Kernel PCA by Davidson: {end-start:<.8f}s elapsed')

    # # Check KPCA results
    # from sklearn.decomposition import KernelPCA
    # kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
    # X_kpca = kpca.fit_transform(X)
    # plt.scatter(X_kpca[y==0,0], X_kpca[y==0,1], color='red')
    # plt.scatter(X_kpca[y==1,0], X_kpca[y==1,1], color='blue')

    plt.scatter(evecs_np[:,0], evecs_np[:,1], color="firebrick", label="numpy")
    plt.scatter(evecs_dav[0], evecs_dav[1], color="orchid", label="davidson")
    plt.legend()
    plt.show()
