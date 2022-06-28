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
    algo_id = 0
    matrices = np.loadtxt(open(file_path, "rb"), delimiter=",")
    titles = [["Config: l= 5, lambda= 1.0e+04, resid= 9.991491e-01, error= 1.201558e-19, iter= 7, time= 0.11",
"Config: l= 5, lambda= 1.0e+02, resid= 7.678201e-01, error= 9.306596e-14, iter= 57, time= 1.03",
"Config: l= 5, lambda= 1.0e+00, resid= 3.910228e-02, error= 1.012390e-08, iter= 1000, time= 15.72",
"Config: l= 5, lambda= 1.0e-02, resid= 3.917733e-04, error= 1.932545e-05, iter= 963, time= 14.54",
"Config: l= 5, lambda= 1.0e-04, resid= 6.890539e-05, error= 1.757153e+00, iter= 1000, time= 18.03",
"Config: l= 10, lambda= 1.0e+04, resid= 9.991491e-01, error= 1.201558e-19, iter= 7, time= 0.11",
"Config: l= 10, lambda= 1.0e+02, resid= 7.678201e-01, error= 2.635659e-14, iter= 23, time= 0.37",
"Config: l= 10, lambda= 1.0e+00, resid= 3.910228e-02, error= 2.754078e-10, iter= 394, time= 5.85",
"Config: l= 10, lambda= 1.0e-02, resid= 3.917733e-04, error= 2.731865e-06, iter= 462, time= 6.87",
"Config: l= 10, lambda= 1.0e-04, resid= 1.831870e-04, error= 5.826962e+00, iter= 1000, time= 17.46",
"Config: l= 15, lambda= 1.0e+04, resid= 9.991491e-01, error= 1.201558e-19, iter= 7, time= 0.11",
"Config: l= 15, lambda= 1.0e+02, resid= 7.678201e-01, error= 2.198338e-14, iter= 22, time= 0.37",
"Config: l= 15, lambda= 1.0e+00, resid= 3.910228e-02, error= 1.579172e-10, iter= 246, time= 4.07",
"Config: l= 15, lambda= 1.0e-02, resid= 3.917733e-04, error= 3.096434e-06, iter= 355, time= 5.80",
"Config: l= 15, lambda= 1.0e-04, resid= 3.511251e-05, error= 1.110167e+00, iter= 514, time= 9.71",
"Config: l= 20, lambda= 1.0e+04, resid= 9.991491e-01, error= 1.201558e-19, iter= 7, time= 0.12",
"Config: l= 20, lambda= 1.0e+02, resid= 7.678201e-01, error= 5.923229e-17, iter= 22, time= 0.33",
"Config: l= 20, lambda= 1.0e+00, resid= 3.910228e-02, error= 1.602522e-10, iter= 139, time= 2.07",
"Config: l= 20, lambda= 1.0e-02, resid= 3.917733e-04, error= 3.574270e-06, iter= 222, time= 3.35",
"Config: l= 20, lambda= 1.0e-04, resid= 1.170137e-04, error= 3.720827e+00, iter= 1000, time= 20.67"],
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
    for i in range(matrices.shape[0]):
        plt.figure(figsize=(10, 4), dpi=80)
        errors = matrices[i, :]
        errors = errors[errors != -1]
        plt.plot([i for i in range(len(errors))], np.log10(errors), label=algo_type[algo_id])
        linear = np.log10([errors[0]/pow(2, i) for i in range(50)])
        quadratic = np.log10([errors[0]]+[errors[0] / pow(2, 2 ** i) for i in range(7)])
        plt.plot(linear, label="linear")
        plt.plot(quadratic, label="quadratic")
        plt.legend()
        plt.xlabel("iterations")
        plt.ylabel("Log error")
        plt.title(algo_type[algo_id]+"\n"+titles[algo_id][i])
        plt.grid(True)
        plt.savefig("./plots/bls/{}_l{}_lambd{}.png".format(algo_type[algo_id], l[int(np.floor(i/5))], lambdas[i % 5]))
        # plt.show()
        plt.clf()


plot_from_error_matrix("testgrid/wolfe_errors_config_jp.csv")
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

