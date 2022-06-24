import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Script that plot the error and residual for ArmijoWolf and BLS,
with different lambdas. It only needs the folder path containing all the data.
"""


def error_residual_plot(name, errdfs, resdfs, lamda_index):
    """
    This method plot the error e residual in log scale for a given lambda index.
    :param name: Name of the algorithm to be displayed in the title.
    :param errdfs: list of couples (l,dataframe) for each type of l.
    :param resdfs: list of couples (l,dataframe) for each type of l.
    :param lamda_index: index of the lambda value, it must be 0<=lamda_index<=3
    :return:
    """
    fig, (ax1, ax2) = plt.subplots(1, 2)
    global lambdas
    fig.suptitle('{} Residual and Error plot with lambda={}'.format(name, lambdas[lamda_index]))
    for (l, df) in errdfs:
        values = df[df[lamda_index] != -1][lamda_index].values
        values = np.log(values)
        ax1.plot([i for i in range(len(values))], values, label="l={}".format(l))
    for (l, df) in resdfs:
        values = df[df[lamda_index] != -1][lamda_index].values
        values = np.log(values)
        ax2.plot([i for i in range(len(values))], values, label="l={}".format(l))
    ax1.set_title('Error plot')
    ax2.set_title('Residual plot')
    ax1.set(xlabel='iterations', ylabel='Log error')
    ax2.set(xlabel='iterations', ylabel='Log residual')
    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")
    plt.show()


def get_l_from_filename(filename):
    """
    Method for obtaining the l number from the filename.
    :param filename:
    :return:
    """
    return filename.split("_")[3]


def plot_from_error_matrix(file_path):
    # matrices = pd.read_csv(file_path).values
    algo_type = ["Armijo-Wolfe", "BLS"]
    algo_id = 1
    matrices = np.loadtxt(open(file_path, "rb"), delimiter=",")
    titles = [["Config: l= 5, lambda= 1.0e+04, error= 2.956111e-19, iter= 8, time= 0.43",
"Config: l= 5, lambda= 1.0e+02, error= 2.467109e-15, iter= 306, time= 7.20",
"Config: l= 5, lambda= 1.0e+00, error= 7.881368e-12, iter= 670, time= 55.78",
"Config: l= 5, lambda= 1.0e-02, error= 1.009371e-06, iter= 589, time= 60.09",
"Config: l= 5, lambda= 1.0e-04, error= 9.880082e-03, iter= 731, time= 57.00",
"Config: l= 10, lambda= 1.0e+04, error= 2.956111e-19, iter= 8, time= 0.17",
"Config: l= 10, lambda= 1.0e+02, error= 1.042756e-15, iter= 298, time= 9.04",
"Config: l= 10, lambda= 1.0e+00, error= 1.529072e-12, iter= 432, time= 59.73",
"Config: l= 10, lambda= 1.0e-02, error= 7.559404e-09, iter= 541, time= 52.50",
"Config: l= 10, lambda= 1.0e-04, error= 7.956725e-03, iter= 505, time= 52.96",
"Config: l= 15, lambda= 1.0e+04, error= 2.956111e-19, iter= 8, time= 0.18",
"Config: l= 15, lambda= 1.0e+02, error= 4.300810e-16, iter= 295, time= 9.63",
"Config: l= 15, lambda= 1.0e+00, error= 2.610516e-11, iter= 110, time= 13.14",
"Config: l= 15, lambda= 1.0e-02, error= 8.728194e-09, iter= 264, time= 19.57",
"Config: l= 15, lambda= 1.0e-04, error= 6.433686e-05, iter= 546, time= 37.51",
"Config: l= 20, lambda= 1.0e+04, error= 2.956111e-19, iter= 8, time= 0.20",
"Config: l= 20, lambda= 1.0e+02, error= 5.194516e-16, iter= 296, time= 9.56",
"Config: l= 20, lambda= 1.0e+00, error= 1.437763e-10, iter= 66, time= 8.74",
"Config: l= 20, lambda= 1.0e-02, error= 1.483125e-08, iter= 122, time= 13.32",
"Config: l= 20, lambda= 1.0e-04, error= 1.552257e-02, iter= 96, time= 13.64"],
              ["Config: l= 5, lambda= 1.0e+04, error= 2.363891e-19, iter= 7, time= 0.09",
"Config: l= 5, lambda= 1.0e+02, error= 4.331375e-11, iter= 62, time= 0.61",
"Config: l= 5, lambda= 1.0e+00, error= 7.013071e-07, iter= 566, time= 5.20",
"Config: l= 5, lambda= 1.0e-02, error= 1.291312e-04, iter= 722, time= 7.44",
"Config: l= 5, lambda= 1.0e-04, error= 3.816404e+01, iter= 222, time= 2.08",
"Config: l= 10, lambda= 1.0e+04, error= 2.363891e-19, iter= 7, time= 0.07",
"Config: l= 10, lambda= 1.0e+02, error= 1.002460e-11, iter= 43, time= 0.42",
"Config: l= 10, lambda= 1.0e+00, error= 2.436646e-08, iter= 329, time= 3.22",
"Config: l= 10, lambda= 1.0e-02, error= 2.749345e-06, iter= 382, time= 4.50",
"Config: l= 10, lambda= 1.0e-04, error= 3.816404e+01, iter= 141, time= 1.76",
"Config: l= 15, lambda= 1.0e+04, error= 2.363891e-19, iter= 7, time= 0.08",
"Config: l= 15, lambda= 1.0e+02, error= 4.074650e-14, iter= 43, time= 0.57",
"Config: l= 15, lambda= 1.0e+00, error= 1.103407e-07, iter= 216, time= 2.55",
"Config: l= 15, lambda= 1.0e-02, error= 1.250372e-05, iter= 348, time= 3.62",
"Config: l= 15, lambda= 1.0e-04, error= 3.816404e+01, iter= 126, time= 1.33",
"Config: l= 20, lambda= 1.0e+04, error= 2.363891e-19, iter= 7, time= 0.08",
"Config: l= 20, lambda= 1.0e+02, error= 1.728218e-13, iter= 34, time= 0.40",
"Config: l= 20, lambda= 1.0e+00, error= 4.434701e-08, iter= 126, time= 1.33",
"Config: l= 20, lambda= 1.0e-02, error= 2.788019e-06, iter= 191, time= 2.00",
"Config: l= 20, lambda= 1.0e-04, error= 3.816404e+01, iter= 84, time= 0.92"]]
    lambdas = ["1e4", "1e2", "1", "1e-2", "1e-4"]
    l = [5, 10, 15, 20]
    plt.figure(figsize=(7, 4), dpi=80)
    for i in range(matrices.shape[0]):
        errors = matrices[i, :]
        plt.plot([i for i in range(len(errors[errors != -1]))], np.log10(errors[errors != -1]))
        plt.xlabel("iterations")
        plt.ylabel("Log error")
        plt.title(algo_type[algo_id]+"\n"+titles[algo_id][i])
        plt.grid(True)
        plt.savefig("./plots/bls/{}_l{}_lambd{}.png".format(algo_type[algo_id], l[int(np.floor(i/5))], lambdas[i % 5]))
        plt.clf()
        # plt.show()
    #for i in range(matrices.shape[0]):
        #errors = matrices[i, :]




plot_from_error_matrix("bls_errors_config.csv")
print("done")

"""folder_path = "results2/"
filenames = os.listdir(folder_path)
wolf_err_list = []
wolf_res_list = []
bls_err_list = []
bls_res_list = []
for f in filenames:
    df = pd.read_csv("{}{}".format(folder_path, f), header=None)
    if "wolf" in f:
        if "err" in f:
            wolf_err_list.append((get_l_from_filename(f), df))
        else:
            wolf_res_list.append((get_l_from_filename(f), df))
    else:
        if "err" in f:
            bls_err_list.append((get_l_from_filename(f), df))
        else:
            bls_res_list.append((get_l_from_filename(f), df))

lambdas = [1, 1e-2, 1e-4, 1e-8]
lamda_index = 3

error_residual_plot("ArmijoWolfe", wolf_err_list, wolf_res_list, lamda_index=lamda_index)
error_residual_plot("BLS", bls_err_list, bls_res_list, lamda_index=lamda_index)"""

