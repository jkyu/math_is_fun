import numpy as np

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
            max_iter=100,
            balanced=False,
            ):

       self.n_guess_per_root = n_guess_per_root
       self.r_threshold = r_threshold # residual 2-norm
       self.max_iter = max_iter
       self.balanced = balanced

    def subspace_preconditioner(self):
        # convergence can be accelerated by an appropriate
        # preconditioning scheme
        # this is usually physically motivated, e.g.,
        # the use of Koopman energies for a CIS Davidson
        pass

    def form_initial_subspace(self, dim, n_guess):
        """Form initial subspace from n_guess unit vectors"""
        # this is a prior that assumes diagonally dominant
        # structure for the matrix of interest
        # easy but not always the right or best choice
        V = np.zeros((dim, n_guess))
        for i in range(n_guess):
            V[i,i] += 1

        return V
    
    def iterate(self, A, V, n_roots):
        """Iterative routine for the Davidson solver"""
        for i in range(self.max_iter):
            # orthogonalize subspace
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
            
            # obtain the residual for each desired root
            V2 = np.copy(V)
            residual_norms = []
            for j in range(n_roots):
                sigma = np.dot(A, np.dot(V, U[:,j]))
                r = sigma - (d[j] * np.dot(V, U[:,j]))
                residual_norms.append(np.linalg.norm(r))
                V2 = np.column_stack((V2, r))

            V = V2
    
            if self.is_converged(residual_norms[:n_roots], i):
                return d[:n_roots], U[:,:n_roots]
    
        raise Exception('Davidson iteration limit reached')

    def orthonormalize(self, V):
        """Orthonormalize subspace basis vectors using QR decomposition"""
        q, r = np.linalg.qr(V)
        return q

    def orthogonalize(self, V):
        # Implement gram-schmidt without normalization
        pass

    def is_converged(self, residual_norms, iteration):
        """Check for convergence and print iteration history"""
        print(f'Iteration {iteration} residual norms: ')
        for i in range(len(residual_norms)):
            print(f'  root {i}: {residual_norms[i]:>.8f}')

        if max(residual_norms) < self.r_threshold:
            return True
        else:
            return False

    def find_roots(self, A, n_roots):
        """Main driver for the Davidson solver"""
        dim = np.shape(A)[0]
        n_guess = n_roots * self.n_guess_per_root
        V = self.form_initial_subspace(dim, n_guess)
        evals, evecs = self.iterate(A, V, n_roots)

        return evals, evecs


def build_rank_sparse_matrix(dim, rank, seed=73):
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
    B = 0.5 * (B + B.T)

    return B


if __name__=='__main__':

    dim = 3000
    rank = 10
    A = build_rank_sparse_matrix(dim, rank)
    
    n_roots = 3
    import time

    # Find the lowest three evals and their evecs 
    # using the Davidson procedure
    start = time.time()
    davidson = Davidson()
    d, U = davidson.find_roots(A, n_roots)
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
    
