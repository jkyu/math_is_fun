import numpy as np

class Vector:
    """
    Class object for each approximate eigenvector,
    referenced herein as a Davidson vector.
    Stores the Davidson vector, its residual, and
    the Davidson eigenvalue. 
    """
    def __init__(
            self,
            vector=None,
            residual=None,
            eigenvalue=None,
            converged=False,
            ):

        self.vector = vector
        self.residual = residual
        self.eigenvalue = eigenvalue
        self.converged = converged

    @staticmethod
    def _l2_norm(vector):
        """l_2 norm of provided"""
        return np.linalg.norm(vector)

    @property
    def residual_norm(self):
        """l_2 norm of the residual vector"""
        return self._l2_norm(self.residual)

    @property
    def _has_vector(self):
        """check if the vector and residual fields have been set"""
        has_vector = isinstance(self.vector, np.ndarray)
        has_residual = isinstance(self.residual, np.ndarray)
        return has_vector and has_residual

    def update(self, new_vector, new_residual, new_eigval, threshold):

        # if self.vector and self.residual are not populated,
        # set them here
        # if self.vector is not yet or no longer converged,
        # update the vector and residual here
        # do nothing (freeze the vectors) if they remain converged
        if not self._has_vector or not self.converged:
            self.vector = new_vector
            self.residual = new_residual
            self.eigenvalue = new_eigval

        # update the new vector if the vector
        # is not converged
        # this includes the case where the vector was
        # converged in a previous iteration but falls
        # out of convergence
        self.converged = np.linalg.norm(new_residual) < threshold


class Davidson:
    """
    Conventions for variable names as corresponding to report document:
        - A is the matrix of interest for which we would like
          to find low-lying eigenvectors/eigenvalues
        - V is the subspace to which we project matrix to solve
          for its eigenvectors/eigenvalues
        - d is a list of all eigenvalues in the context its used
        - U is a unitary matrix whose columns are eigenvectors
    """
    def __init__(
            self,
            n_guess_per_root=2,
            res_threshold=1e-6,
            max_iter=200,
            balanced=False,
            ):

        self.n_guess_per_root = n_guess_per_root
        self.res_threshold = res_threshold # residual 2-norm
        self.max_iter = max_iter
        self.balanced = balanced
        # self.vectors will be set when find_roots() is called
        self.vectors = None 

    def precondition(self, matrix, residual, eigval):
        # convergence can be accelerated by an appropriate
        # preconditioning scheme
        # the standard preconditioner for implementations 
        # of the Davidson procedure in EST codes uses
        # (D - eigval*I)^-1 as a preconditioner to the 
        # trial vector r, where D is the diagonal
        # of the starting matrix matrix
        # this is best used for diagonally dominant
        # matrices -- won't work as well for rank sparse 
        # but spatially dense matrices 
        diag = np.diag(matrix)
        diag_precond = 1 / (np.diag(matrix) - eigval)
        precond = np.diag(diag_precond)
        precond_residual = np.dot(precond, residual)

        return precond_residual

    def form_initial_subspace(self, dim, n_guess):
        """Form initial subspace from n_guess unit vectors"""
        # this is a prior that assumes diagonally dominant
        # structure for the matrix of interest
        # easy but not always the right or best choice
        # submat stands for subspace matrix
        submat = np.zeros((dim, n_guess))
        for i in range(n_guess):
            submat[i,i] += 1

        return submat
    
    def iterate(self, matrix, submat, n_roots, precondition=False, verbose=False):
        """Iterative routine for the Davidson solver"""
        if verbose:
            print('#=> Begin Davidson procedure')

        for i in range(self.max_iter):
            # orthogonalize subspace except for the first 
            # iteration
            if i > 0:
                if self.balanced:
                    # not yet implemented
                    submat = self.orthogonalize(submat)
                else:
                    submat = self.orthonormalize(submat)

            # project matrix A into the subspace spanned by V
            # note: this step is actually not done explicitly 
            # in real codes, where the matrix-vector product
            # sigma = AVu is formed directly in the subspace
            projmat = np.dot(np.dot(submat.T, matrix), submat)
            
            # solve the subspace eigenvalue problem
            # conventional notation for eigenvalue problems:
            # U = unitary matrix
            diag, U = np.linalg.eigh(projmat)
            idx = diag.argsort()
            diag = diag[idx]
            U = U[:,idx]
            
            # make a copy of the subspace matrix 
            # in order to expand it if necessary
            # without changing it when treating multiple roots
            submat_expanded = np.copy(submat)

            # obtain the residual for each desired root
            for j, dav_vec in enumerate(self.vectors):
                new_eigval = diag[j]
                new_eigvec = U[:, j]

                sigma = np.dot(matrix, np.dot(submat, new_eigvec))
                residual = sigma - (new_eigval * np.dot(submat, new_eigvec))

                dav_vec.update(
                        sigma / np.linalg.norm(sigma), 
                        residual, 
                        new_eigval, 
                        self.res_threshold
                        )

                # only expand the subsace for Davidson vectors
                # that are not converged
                if not dav_vec.converged:
                    # precondition if requested
                    # works well for diagonally dominant, spatially
                    # sparse matrices (like in EST)
                    if precondition:
                        residual = self.precondition(matrix, residual, new_eigval)
                    submat_expanded = np.column_stack((submat_expanded, residual))
            # overwrite submat with the new expanded subspace
            submat = submat_expanded
    
            # optional option to print iteration history
            if verbose:
                self.print_convergence(i)

            # return if all Davidson vectors are converged
            if all([v.converged for v in self.vectors]):
                if verbose:
                    print('#=> Davidson procedure converged.')
                return 
    
        print(f'Warning: iteration limit of {self.max_iter} reached')

    def orthonormalize(self, submat):
        """Orthonormalize subspace basis vectors using QR decomposition"""
        q, r = np.linalg.qr(submat)
        return q

    def orthogonalize(self, submat):
        # Implement gram-schmidt without normalization
        pass

    def print_convergence(self, iteration):
        """Print iteration history"""
        print(f'Iteration {iteration} residual norms: ')
        for i, vec in enumerate(self.vectors):
            if vec.converged:
                converged = 'Converged'
            else:
                converged = 'Not Converged'
            print(f'  root {i}: {vec.residual_norm:>.8f} {converged}')

    def find_roots(
            self, 
            matrix, 
            n_roots, 
            largest_roots=False,
            precondition=False, 
            verbose=False
            ):
        """Main driver for the Davidson solver"""
        # instantiate Davidson vectors
        self.vectors = [ Vector() for _ in range(n_roots) ]

        # set up and solve
        dim = np.shape(matrix)[0]
        n_guess = n_roots * self.n_guess_per_root
        submat = self.form_initial_subspace(dim, n_guess)

        # if we want the largest eigenvalues,
        # find the smallest eigenvalues of the negative matrix
        if largest_roots:
            matrix = -matrix
        self.iterate(matrix, submat, n_roots, 
                precondition=precondition, 
                verbose=verbose
                )

        # return converged eigenvectors and eigenvalues
        eigvals = [ v.eigenvalue for v in self.vectors ]
        scaler = -1 if largest_roots else 1
        eigvecs = [ scaler * v.vector for v in self.vectors ]

        return eigvals, eigvecs


def rank_sparse_matrix(dim, rank, noise=1e-3, seed=73):
    """Build rank-sparse matrix for testing the Davidson routine"""
    # set random seed for reproducibility
    np.random.seed(seed)

    # get orthonormal basis via QR
    matrix = np.zeros((dim,dim))
    for i in range(dim):
        matrix[:,i] = np.random.randn(dim)
    q, r = np.linalg.qr(matrix)

    # construct a hermitian matrix of some rank
    # will be rank sparse if rank << dim
    assert rank <= dim

    evals = np.random.randn(rank)
    evals.sort()
    # print('Reference evals: ', evals)

    sparsemat = np.zeros((dim,dim))
    for i in range(rank):
        sparsemat += evals[i] * np.outer(q[:,i], q[:,i])
    sparsemat += noise*np.random.randn(dim,dim)
    sparsemat = 0.5 * (sparsemat + sparsemat.T)

    return sparsemat

def diagonally_dominant_matrix(dim, noise=1e-3, seed=73):

    matrix = np.eye(dim)
    matrix += np.random.randn(dim,dim)
    matrix = 0.5 * (matrix + matrix.T)

    return matrix


if __name__=='__main__':

    import time
    dim = 1000
    rank = 100
    n_roots = 10

    start = time.time()
    matrix = rank_sparse_matrix(dim, rank)
    end = time.time()
    print(f'Setup: {end-start:<.8f}s elapsed')

    # Find the lowest three evals and their evecs 
    # using the Davidson procedure
    start = time.time()
    davidson = Davidson()
    d, U = davidson.find_roots(
            matrix, 
            n_roots, 
            largest_roots=False,
            precondition=False, 
            verbose=True
            )
    print(d)
    end = time.time()
    print(f'Davidson: {end-start:<.8f}s elapsed')
    
    # Find all eval/evecs simultaneously by
    # explicit diagonalization
    start = time.time()
    d2, U2 = np.linalg.eigh(matrix)
    idx = np.argsort(d2)
    print(d2[idx[:n_roots]])
    d2 = d2[idx[::-1]]
    print(d2[:n_roots])
    end = time.time()
    print(f'NumPy linalg: {end-start:<.8f}s elapsed')

