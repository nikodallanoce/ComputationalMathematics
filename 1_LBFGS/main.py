import scipy as sp
from scipy.optimize import minimize
from scipy.linalg import lstsq
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

list_xk = []


def f_lls(w: np.ndarray, *args) -> float:
    arg = args[0]
    XtX: np.ndarray = arg[0]
    ytX2: np.ndarray = arg[1]
    yty: np.ndarray = arg[2]
    f_lls = w.transpose().dot(XtX).dot(w) - ytX2.dot(w) + yty
    return f_lls


def grad_lls(w, *args) -> np.ndarray:
    arg = args[0]
    XtX: np.ndarray = arg[0]
    ytX2: np.ndarray = arg[1]
    return 2 * w.transpose().dot(XtX) - ytX2


def build_matrix_y(lam: float):
    X_hat = np.loadtxt(open("ML-CUP21-TR.csv", "rb"), delimiter=",", skiprows=7)
    X_hat: np.ndarray = X_hat[:, 1:]
    m, n0 = X_hat.shape
    # id = np.identity(m)*lam
    X_hat = np.concatenate((X_hat.transpose(), np.identity(m) * lam), 0)
    m, n = X_hat.shape
    # X_hat = np.random.rand(500, 20)
    y_hat = np.concatenate((np.random.rand(n0), np.zeros(m - n0)), 0)
    return (X_hat, y_hat)


def load_matrix_y_w(X_path: str, y_path: str, w_path: str):
    X_hat = scipy.io.loadmat(X_path)['X_hat']
    y_hat = scipy.io.loadmat(y_path)['y_hat']
    y_hat = np.reshape(y_hat, (y_hat.shape[0]))
    w = scipy.io.loadmat(w_path)['w']
    w = np.reshape(w, (w.shape[0]))
    return X_hat, y_hat, w


def call(xk) -> bool:
    list_xk.append(xk)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    np.random.seed(0)
    X_hat, y_hat = build_matrix_y(1e2)
    w = np.random.rand(X_hat.shape[1])
    #X_hat, y_hat, w = load_matrix_y_w("X_hat.mat", "y_hat.mat", "w.mat")
    XtX: np.ndarray = X_hat.transpose().dot(X_hat)
    ytX2: np.ndarray = 2 * y_hat.transpose().dot(X_hat)
    yty: np.ndarray = y_hat.transpose().dot(y_hat)
    arguments = [XtX, ytX2, yty]
    f_ris = f_lls(w, (XtX, ytX2, yty))
    g_ris = grad_lls(w, (XtX, ytX2))
    tol = 1e-12
    opt_bfgs = {'gtol': 1e-12, 'disp':True}
    opt_lbfgs = {'disp': True, 'maxcor': 1, 'ftol': 1e-12, 'gtol': 1e-12, 'eps': 1e-12, 'maxfun': 15000,
                 'maxiter': 15000, 'iprint': -1, 'maxls': 20, 'finite_diff_rel_step': None}
    ris = minimize(fun=f_lls, x0=w, args=arguments, method='L-BFGS-B', jac=grad_lls, tol=tol, options=opt_lbfgs,
                   callback=call)

    w_bfgs = ris['x']
    w_star, res, rnk, s = lstsq(X_hat, y_hat)
    ris_np = np.linalg.lstsq(X_hat, y_hat, rcond=None)
    norm = np.linalg.norm(w_star - w_bfgs)

    print("{0:1.16e}".format(norm))
    print("{0:1.5e}".format(np.linalg.cond(X_hat)))
    print("{0:1.16e}".format(np.linalg.norm(X_hat.dot(w_star) - y_hat)))

    error = np.zeros(len(list_xk))
    p_values = np.zeros(len(list_xk) - 1)
    f_star = f_lls(w_star, arguments)
    f_val_last = 0
    for i, x_k in enumerate(list_xk):
        error[i] = np.linalg.norm(x_k - w_star)
        f_val_k = f_lls(x_k, arguments)
        if (i != 0):
            p_values[i - 1] = np.log(abs(f_val_k - f_star)) / np.log(abs(f_val_last - f_star))
            print(p_values[i-1])
        f_val_last = f_val_k

    plt.plot(p_values)
    plt.show()

    print()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
