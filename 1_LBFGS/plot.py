import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import genfromtxt

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
    algo_type = ["Our L-BFGS", "BLS"]
    algo_id = 0
    matrices = np.loadtxt(open(file_path, "rb"), delimiter=",")
    titles = [["Config: l= 5, lambda= 1.0e+04",
"Config: l= 5, lambda= 1.0e+02",
"Config: l= 5, lambda= 1.0e+00",
"Config: l= 5, lambda= 1.0e-02",
"Config: l= 5, lambda= 1.0e-04",
"Config: l= 10, lambda= 1.0e+04",
"Config: l= 10, lambda= 1.0e+02",
"Config: l= 10, lambda= 1.0e+00",
"Config: l= 10, lambda= 1.0e-02",
"Config: l= 10, lambda= 1.0e-04",
"Config: l= 15, lambda= 1.0e+04",
"Config: l= 15, lambda= 1.0e+02",
"Config: l= 15, lambda= 1.0e+00",
"Config: l= 15, lambda= 1.0e-02",
"Config: l= 15, lambda= 1.0e-04",
"Config: l= 20, lambda= 1.0e+04",
"Config: l= 20, lambda= 1.0e+02",
"Config: l= 20, lambda= 1.0e+00",
"Config: l= 20, lambda= 1.0e-02",
"Config: l= 20, lambda= 1.0e-04"],
              []]
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
        if i > 14:
            scipy_data = genfromtxt('scipy_errors{}.csv'.format((i % 5)), delimiter=',')
            scipy_data[0] = errors[0]
            plt.plot(np.log10(scipy_data), label="Scipy L-BFGS")
        plt.legend()
        plt.xlabel("iterations")
        plt.ylabel("Log error")
        # plt.title("Armijo-Wolfe\n"+titles[algo_id][i])
        plt.title("Comparison with an off-the-shelf solver\n" + titles[algo_id][i])
        plt.grid(True)
        # plt.savefig("./plots/wolfe/aw_l{}_lambda{}.png".format(l[int(np.floor(i/5))], lambdas[i % 5]))
        plt.show()
        # plt.clf()


def mean_std_errors(file_paths):
    run1 = np.loadtxt(open(file_paths[0], "rb"), delimiter=",")
    run2 = np.loadtxt(open(file_paths[1], "rb"), delimiter=",")
    run3 = np.loadtxt(open(file_paths[2], "rb"), delimiter=",")
    mat = np.zeros((20, 3))
    for j in range(run1.shape[0]):
        vect = run1[j, :]
        vect = vect[vect != -1]
        mat[j, 0] = vect[-2]
    for j in range(run2.shape[0]):
        vect = run2[j, :]
        val = vect[vect != -1][-2]
        mat[j, 1] = val
    for j in range(run3.shape[0]):
        vect = run3[j, :]
        val = vect[vect != -1][-2]
        mat[j, 2] = val
    print(mat)
    for l in mat:
        print("mean:{0:.2e} std:{1:.2e}".format(np.mean(l), np.std(l)))


def var_in_lambda(file_path):
    matrices = np.loadtxt(open(file_path, "rb"), delimiter=",")
    titles = [["Config: l= 5, lambda= 1.0e+04, resid= 9.995807e-01, error= 1.673263e-14, iter= 7, time= 0.04",
"Config: l= 5, lambda= 1.0e+02, resid= 8.189265e-01, error= 5.182123e-14, iter= 66, time= 0.34",
"Config: l= 5, lambda= 1.0e+00, resid= 4.658075e-02, error= 1.060092e-12, iter= 512, time= 2.67",
"Config: l= 5, lambda= 1.0e-02, resid= 4.067113e-04, error= 7.924396e-13, iter= 519, time= 3.00",
"Config: l= 5, lambda= 1.0e-04, resid= 5.085703e-06, error= 1.016759e-12, iter= 452, time= 2.48",
"Config: l= 10, lambda= 1.0e+04, resid= 9.997820e-01, error= 3.032756e-14, iter= 11, time= 0.06",
"Config: l= 10, lambda= 1.0e+02, resid= 9.476044e-01, error= 5.586512e-14, iter= 31, time= 0.17",
"Config: l= 10, lambda= 1.0e+00, resid= 5.837749e-02, error= 5.487410e-13, iter= 318, time= 1.65",
"Config: l= 10, lambda= 1.0e-02, resid= 5.308875e-04, error= 7.301883e-13, iter= 409, time= 2.11",
"Config: l= 10, lambda= 1.0e-04, resid= 6.321317e-06, error= 1.095120e-12, iter= 362, time= 1.87",
"Config: l= 15, lambda= 1.0e+04, resid= 9.998782e-01, error= 3.207052e-14, iter= 16, time= 0.09",
"Config: l= 15, lambda= 1.0e+02, resid= 8.920362e-01, error= 6.040599e-14, iter= 22, time= 0.12",
"Config: l= 15, lambda= 1.0e+00, resid= 5.294190e-02, error= 7.013250e-13, iter= 186, time= 0.98",
"Config: l= 15, lambda= 1.0e-02, resid= 4.209989e-04, error= 3.192818e-13, iter= 207, time= 1.10",
"Config: l= 15, lambda= 1.0e-04, resid= 6.451557e-06, error= 6.100493e-13, iter= 251, time= 1.33",
"Config: l= 20, lambda= 1.0e+04, resid= 9.998875e-01, error= 2.553744e-14, iter= 21, time= 0.12",
"Config: l= 20, lambda= 1.0e+02, resid= 9.461350e-01, error= 7.180798e-15, iter= 21, time= 0.12",
"Config: l= 20, lambda= 1.0e+00, resid= 6.065269e-02, error= 2.524833e-13, iter= 134, time= 0.71",
"Config: l= 20, lambda= 1.0e-02, resid= 5.019611e-04, error= 1.465840e-13, iter= 78, time= 0.42",
"Config: l= 20, lambda= 1.0e-04, resid= 4.620518e-06, error= 3.538826e-13, iter= 165, time= 0.88"], []]
    plt.figure(figsize=(10, 4), dpi=80)
    indices = [1, 6, 11, 16]
    l = [5, 10, 15, 20]
    for l_ind, i in enumerate(indices):
        errors = matrices[i, :]
        errors = errors[errors != -1]
        plt.plot([i for i in range(len(errors))], np.log10(errors), label="l={}".format(l[l_ind]))
    linear = np.log10([errors[0] / pow(2, i) for i in range(50)])
    quadratic = np.log10([errors[0]] + [errors[0] / pow(2, 2 ** i) for i in range(7)])
    plt.plot(linear, label="linear")
    plt.plot(quadratic, label="quadratic")
    plt.legend()
    plt.xlabel("iterations")
    plt.ylabel("Log error")
    plt.title("Comparison using different l values\n" + "lambda=1e2")
    plt.grid(True)
    plt.show()


# mean_std_errors(['residues1.csv', 'residues2.csv', 'residues3.csv'])
# mean_std_errors(['errors1.csv', 'errors2.csv', 'errors3.csv'])
plot_from_error_matrix("errors2.csv")
# var_in_lambda("errors2.csv")
print("done")

