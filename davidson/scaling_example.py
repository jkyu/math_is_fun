from davidson import Davidson, rank_sparse_matrix
from kpca import run_kpca
from sklearn.datasets import make_moons
import time
import numpy as np


def davidson_scaling(rank=10, n_roots=3):

    davidson = Davidson()
    for dim in [10, 100, 1000, 5000, 10000]:
        print(f"> Dimension {dim}; rank {rank}")
        A = rank_sparse_matrix(dim, rank)
        start = time.time()
        evals_d, evecs_d = davidson.find_roots(
                A,
                n_roots=n_roots,
                largest_roots=True,
                precondition=False,
                verbose=False
                )
        end = time.time()
        print(f"Davidson time: {end-start:>.4}s")

        start = time.time()
        evals_n, evecs_n = np.linalg.eigh(A)
        evals_n = [evals_n[-i] for i in range(1,n_roots+1)]
        evecs_n = [evecs_n[:,-i] for i in range(1,n_roots+1)]
        end = time.time()
        print(f"NumPy time: {end-start:>.4}s")

        for i in range(n_roots):
            eval_error = np.abs(evals_d[i] - evals_n[i])
            # evec_error == 2 means there's a phase factor
            # evec_error == 0 means no phase factor
            # both are good
            evec_error = np.linalg.norm(evecs_d[i] - evecs_n[i])
            print(f"Root {i} error: ")
            print(f"  eval: {eval_error:>.4}")
            print(f"  evec: {evec_error:>.4}")
        print()

def kpca_scaling():

    for npts in [10, 100, 1000, 5000, 10000]:
        print(f"> Number of points: {npts}")
        data, _ = make_moons(n_samples=npts, random_state=0)
        run_kpca(data, use_davidson=False)
        run_kpca(data, use_davidson=True)
        print()


if __name__=="__main__":
    davidson_scaling()
    kpca_scaling()




