# Davidson Solver

## Summary
Implementation of a basic Davidson routine for finding extreme eigenvalues and their associated eigenvectors without employing a full eigendecomposition. 

A brief background of the Davidson method is provided in `reference/background.pdf` and a simple implementation of the algorithm is in `davidson.py`.

A small demonstration of the Davidson is provided in kpca.py. 
This includes a basic implementation of kernel PCA. 
Normally, a standard eigendecomposition is used to solve for the principal component vectors.
Since our visual perception is usually limited to 1-3D, PCA can be accelerated by avoiding the full eigendecomposition.
On my machine, using the Davidson instead of NumPy's `linalg.eigh` routine nets a ~20x speedup for obtaining the lowest two principal component vectors for a 5000 x 5000 kernel matrix.  
