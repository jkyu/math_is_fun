from davidson import Davidson, rank_sparse_matrix
from kpca import run_kpca
from sklearn.datasets import make_moons
import time
import numpy as np


def davidson_scaling(rank=10, n_roots=3):

    davidson = Davidson()
    dim_list = [10, 100, 500, 1000, 2000]
    dim_list += [3000, 4000, 5000, 6000]
    dim_list += [7000, 8000, 9000, 10000]
    time_n = []
    time_d = []
    max_errors_eval = []
    max_errors_evec = []
    for dim in dim_list:
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
        t_d = end-start
        time_d.append(t_d)
        print(f"Davidson time: {t_d:>.4}s")

        start = time.time()
        evals_n, evecs_n = np.linalg.eigh(A)
        evals_n = [evals_n[-i] for i in range(1,n_roots+1)]
        evecs_n = [evecs_n[:,-i] for i in range(1,n_roots+1)]
        end = time.time()
        t_n = end-start
        time_n.append(t_n)
        print(f"NumPy time: {t_n:>.4}s")

        errors_eval = []
        errors_evec = []
        for i in range(n_roots):
            eval_error = np.abs(evals_d[i] - evals_n[i])
            # evec_error == 2 means there's a phase factor
            # evec_error == 0 means no phase factor
            # both are good
            evec_error = np.linalg.norm(evecs_d[i] - evecs_n[i])
            print(f"Root {i} error: ")
            print(f"  eval: {eval_error:>.4}")
            print(f"  evec: {evec_error:>.4}")
            errors_eval.append(eval_error)
            errors_evec.append(evec_error)
        max_errors_eval.append(max(errors_eval))
        max_errors_evec.append(max(errors_evec))
        print()
    print("Dimension: ", dim_list)
    print("Compute times (NumPy): ", time_n)
    print("Compute times (Davidson): ", time_d)
    print("Maximum eval errors: ", max_errors_eval)
    print("Maximum evec errors: ", max_errors_evec)

def kpca_scaling():

    dim_list = [10, 100, 500, 1000, 2000]
    dim_list += [3000, 4000, 5000, 6000]
    dim_list += [7000, 8000, 9000, 10000]
    for npts in dim_list:
        print(f"> Number of points: {npts}")
        data, _ = make_moons(n_samples=npts, random_state=0)
        run_kpca(data, use_davidson=False)
        run_kpca(data, use_davidson=True)
        print()


if __name__=="__main__":
    # davidson_scaling()
    kpca_scaling()




