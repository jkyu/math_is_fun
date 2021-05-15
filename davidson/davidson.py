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
        has_residual = isinstance(self.vector, np.ndarray)
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
        if np.linalg.norm(new_residual) < threshold:
            self.converged = True
        else:
            self.converged = False


class Davidson:
    """
    Conventions for variable names:
        - A is the matrix of interest for which we would like
          to find low-lying eigenvectors/eigenvalues
        - V is the subspace to which we project A to solve
          for its eigenvectors/eigenvalues
        - d is a list of all eigenvalues in the context its used
        - U is a unitary matrix whose columns are eigenvectors
    """
    def __init__(
            self,
            n_guess_per_root=2,
            r_threshold=1e-6,
            max_iter=200,
            balanced=False,
            ):

        self.n_guess_per_root = n_guess_per_root
        self.r_threshold = r_threshold # residual 2-norm
        self.max_iter = max_iter
        self.balanced = balanced
        # self.vectors will be set when find_roots() is called
        self.vectors = None 
        # store all eigenvalues for all trial vectors
        # for use in the preconditioner
        self.trial_eigvals = None

    def precondition(self, A, r, eigval):
        # convergence can be accelerated by an appropriate
        # preconditioning scheme
        # the standard preconditioner for implementations 
        # of the Davidson procedure in EST codes uses
        # (D - eigval*I)^-1 as a preconditioner to the 
        # trial vector r, where D is the diagonal
        # of the starting matrix A
        # this is best used for diagonally dominant
        # matrices -- won't work as well for rank sparse 
        # but spatially dense matrices 
        diag_A = np.diag(A)
        diag_precond = 1 / (np.diag(A) - eigval)
        precond = np.diag(diag_precond)
        precond_r = np.dot(precond, r)

        return precond_r

    def form_initial_subspace(self, dim, n_guess):
        """Form initial subspace from n_guess unit vectors"""
        # this is a prior that assumes diagonally dominant
        # structure for the matrix of interest
        # easy but not always the right or best choice
        V = np.zeros((dim, n_guess))
        for i in range(n_guess):
            V[i,i] += 1

        return V
    
    def iterate(self, A, V, n_roots, precondition=False, verbose=False):
        """Iterative routine for the Davidson solver"""
        if verbose:
            print('#=> Begin Davidson procedure')

        for i in range(self.max_iter):
            # orthogonalize subspace except for the first 
            # iteration
            if i > 0:
                if self.balanced:
                    # not yet implemented
                    V = self.orthogonalize(V)
                else:
                    V = self.orthonormalize(V)

            # project matrix A into the subspace spanned by V
            # note: this step is actually not done explicitly 
            # in real codes, where the matrix-vector product
            # sigma = AVu is formed directly in the subspace
            B = np.dot(np.dot(V.T, A), V)
            
            # solve the subspace eigenvalue problem
            d, U = np.linalg.eigh(B)
            idx = d.argsort()
            d = d[idx]
            U = U[:,idx]
            
            # make a copy of the subspace matrix V
            # in order to expand it if necessary
            # without changing V when treating multiple roots
            V2 = np.copy(V)

            # obtain the residual for each desired root
            for j, dav_vec in enumerate(self.vectors):
                new_eigval = d[j]
                new_eigvec = U[:, j]

                sigma = np.dot(A, np.dot(V, new_eigvec))
                residual = sigma - (new_eigval * np.dot(V, new_eigvec))

                dav_vec.update(
                        new_eigvec, 
                        residual, 
                        new_eigval, 
                        self.r_threshold
                        )

                # only expand the subsace for Davidson vectors
                # that are not converged
                if not dav_vec.converged:
                    # precondition if requested
                    # works well for diagonally dominant, spatially
                    # sparse matrices (like in EST)
                    if precondition:
                        residual = self.precondition(A, residual, new_eigval)
                    V2 = np.column_stack((V2, residual))

            # overwrite V with the new expanded subspace
            V = V2
    
            # optional option to print iteration history
            if verbose:
                self.print_convergence(i)

            # return if all Davidson vectors are converged
            if all([v.converged for v in self.vectors]):
                if verbose:
                    print('#=> Davidson procedure converged.')
                return 
    
        print(f'Warning: iteration limit of {self.max_iter} reached')

    @staticmethod
    def orthonormalize(V):
        """Orthonormalize subspace basis vectors using QR decomposition"""
        q, r = np.linalg.qr(V)
        return q

    def orthogonalize(self, V):
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

    def find_roots(self, A, n_roots, precondition=False, verbose=False):
        """Main driver for the Davidson solver"""
        # instantiate Davidson vectors
        self.vectors = [ Vector() for _ in range(n_roots) ]

        # set up and solve
        dim = np.shape(A)[0]
        n_guess = n_roots * self.n_guess_per_root
        V = self.form_initial_subspace(dim, n_guess)
        self.iterate(A, V, n_roots, 
                precondition=precondition, 
                verbose=verbose
                )

        # return converged eigenvectors and eigenvalues
        eigvals = [ v.eigenvalue for v in self.vectors ]
        eigvecs = [ v.vector for v in self.vectors ]

        return eigvals, eigvecs


def rank_sparse_matrix(dim, rank, noise=1e-3, seed=73):
    """Build rank-sparse matrix for testing the Davidson routine"""
    # set random seed for reproducibility
    np.random.seed(seed)

    # get orthonormal basis via QR
    A = np.zeros((dim,dim))
    for i in range(dim):
        A[:,i] = np.random.randn(dim)
    q, r = np.linalg.qr(A)

    # construct a hermitian matrix of some rank
    # will be rank sparse if rank << dim
    assert rank <= dim

    evals = np.random.randn(rank)
    evals.sort()
    print('Reference evals: ', evals)

    B = np.zeros((dim,dim))
    for i in range(rank):
        B += evals[i] * np.outer(q[:,i], q[:,i])
    B += noise*np.random.randn(dim,dim)
    B = 0.5 * (B + B.T)

    return B

def diagonally_dominant_matrix(dim, noise=1e-3, seed=73):

    A = np.eye(dim)
    A += np.random.randn(dim,dim)
    A = 0.5 * (A + A.T)

    return A


if __name__=='__main__':

    import time
    dim = 1000
    rank = 100
    n_roots = 5

    start = time.time()
    A = rank_sparse_matrix(dim, rank, noise=1e-1)
    end = time.time()
    print(f'Setup: {end-start:<.8f}s elapsed')

    # Find the lowest three evals and their evecs 
    # using the Davidson procedure
    start = time.time()
    davidson = Davidson()
    d, U = davidson.find_roots(
            A, 
            n_roots, 
            precondition=False, 
            verbose=True
            )
    print(d)
    end = time.time()
    print(f'Davidson: {end-start:<.8f}s elapsed')
    
    # Find all eval/evecs simultaneously by
    # explicit diagonalization
    start = time.time()
    d2, U2 = np.linalg.eigh(A)
    idx = np.argsort(d2)
    print(d2[idx[:n_roots]])
    end = time.time()
    print(f'NumPy linalg: {end-start:<.8f}s elapsed')
    
