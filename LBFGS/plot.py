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


folder_path = "results2/"
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
error_residual_plot("BLS", bls_err_list, bls_res_list, lamda_index=lamda_index)

