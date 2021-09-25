import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({"figure.autolayout": True})

def plot_davidson_scaling():

    dim = [10, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    time_n = [0.00012540817260742188, 0.00341796875, 0.06534910202026367, 0.6627819538116455, 5.3491034507751465, 12.427345752716064, 19.07955574989319, 32.51601958274841, 50.15958046913147, 79.19470477104187, 115.61950325965881, 164.77915930747986, 212.9846966266632]
    time_d = [0.0011277198791503906, 0.00596165657043457, 0.04130840301513672, 0.057979583740234375, 2.374391794204712, 0.9196336269378662, 1.3185594081878662, 5.722989797592163, 2.7151997089385986, 3.021615743637085, 4.066304445266724, 4.65549111366272, 12.284252405166626]

    fig = plt.figure(figsize=(6,5))
    labelsize = 16
    ticksize = 14
    plt.rc("xtick", labelsize=ticksize)
    plt.rc("ytick", labelsize=ticksize)

    plt.plot(dim, time_d, color="firebrick", marker="o", label="Davidson")
    plt.plot(dim, time_n, color="steelblue", marker="o", label="NumPy")

    plt.title("Davidson Scaling", fontsize=labelsize)
    plt.xlabel("Matrix Dimension [$n$]", fontsize=labelsize)
    plt.ylabel("Wall Time [seconds]", fontsize=labelsize)
    plt.legend(loc="best", frameon=False, fancybox=False, fontsize=ticksize, numpoints=1)

    plt.savefig("./report/figures/dav_scaling.pdf", dpi=300)

def plot_davidson_error():

    dim = [10, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    # note that the evec errors will be approximately two if there is an error in phase factor
    eval_error = [1.1102230246251565e-15, 1.730837695390619e-13, 4.4086956307864966e-13, 7.30526750203353e-14, 2.411804089774705e-11, 1.1191048088221578e-13, 1.5765166949677223e-14, 5.572459160774201e-12, 2.879918525877656e-13, 2.5979218776228663e-13, 7.482903185973555e-14, 2.0228263508670352e-13, 4.382022522619877e-12]
    evec_error = [0.0, 0.0, 6.661338147750939e-16, 0.0, 8.881784197001252e-16, 0.0, 4.440892098500626e-16, 4.440892098500626e-16, 4.440892098500626e-16, 1.3322676295501878e-15, 4.440892098500626e-16, 2.0017604393173597e-08, 2.220446049250313e-16]

    fig = plt.figure(figsize=(6,5))
    labelsize = 16
    ticksize = 14
    plt.rc("xtick", labelsize=ticksize)
    plt.rc("ytick", labelsize=ticksize)

    plt.plot(dim, eval_error, color="darkcyan", marker="o", label="Maximum Eigenvalue Error")
    plt.plot(dim, evec_error, color="darkturquoise", marker="o", label="Maximum Eigenvector $l_2$ Error")

    plt.title("Davidson Accuracy", fontsize=labelsize)
    plt.xlabel("Matrix Dimension [$n$]", fontsize=labelsize)
    plt.ylabel("Error", fontsize=labelsize)
    plt.legend(loc="best", frameon=False, fancybox=False, fontsize=ticksize, numpoints=1)

    plt.savefig("./report/figures/dav_error.pdf", dpi=300)

def plot_kpca_scaling():

    dim = [10, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    time_K = [0.0006, 0.0503, 1.1895, 4.5962, 18.0422, 40.7698, 72.1031, 113.2667, 162.0825, 219.8207, 289.1012, 365.8915, 452.9805]
    time_eig_d = [0.0002, 0.003, 0.0409, 0.2316, 2.5757, 5.3745, 12.7833, 23.5044, 41.6721, 64.2073, 97.6045, 135.2819, 187.4736]
    time_eig_n = [0.001, 0.0141, 0.0395, 0.1015, 0.4377, 0.7751, 1.6213, 2.2132, 3.5388, 4.707, 6.7647, 7.3962, 10.3599]
    time_tot_d = [ x+y for x,y in zip(time_K, time_eig_d) ]
    time_tot_n = [ x+y for x,y in zip(time_K, time_eig_n) ]

    fig = plt.figure(figsize=(6,5))
    labelsize = 16
    ticksize = 14
    plt.rc("xtick", labelsize=ticksize)
    plt.rc("ytick", labelsize=ticksize)

    plt.plot(dim, time_tot_d, color="firebrick", marker="o", label="KPCA total (Davidson)")
    plt.plot(dim, time_eig_d, color="firebrick", marker="x", linestyle="--", label="Eigensolve (Davidson)")
    plt.plot(dim, time_tot_n, color="steelblue", marker="o", label="KPCA total (NumPy)")
    plt.plot(dim, time_eig_n, color="steelblue", marker="x", linestyle="--", label="Eigensolve (NumPy)")
    plt.plot(dim, time_K, color="black", marker="*", linestyle=":", label="Kernel matrix build")

    plt.title("Kernel PCA Scaling", fontsize=labelsize)
    plt.xlabel("Matrix Dimension [$n$]", fontsize=labelsize)
    plt.ylabel("Wall Time [seconds]", fontsize=labelsize)
    plt.legend(loc="best", frameon=False, fancybox=False, fontsize=ticksize, numpoints=1)

    plt.savefig("./report/figures/kpca_scaling.pdf", dpi=300)

plot_davidson_scaling()
plot_davidson_error()
plot_kpca_scaling()
