import scipy as sp
from numpy.linalg import cond
from scipy.optimize import minimize
from scipy.linalg import lstsq
from scipy.optimize import fmin_bfgs, fmin_l_bfgs_b
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

list_xk = []
list_xk = []
name = "L-BFGS Scipy"
lambdas = [1e4, 1e2, 1, 1e-2, 1e-4]

def f_lls(w: np.ndarray, *args) -> float:
    arg = args[0]
    X: np.ndarray = arg[0]
    ytX2: np.ndarray = arg[1]
    yty: np.ndarray = arg[2]
    Xw = X.dot(w)
    f_lls = Xw.T.dot(Xw) - ytX2.dot(w) + yty
    return f_lls


def grad_lls(w, *args) -> np.ndarray:
    arg = args[0]
    X: np.ndarray = arg[0]
    ytX2: np.ndarray = arg[1]
    return 2 * X.dot(w).T.dot(X) - ytX2


def build_matrix_y(lam: float):
    X_hat = np.loadtxt(open("../datasets/ML-CUP21-TR.csv", "rb"), delimiter=",", skiprows=7)
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
    #np.random.seed(2)
    scipy_errors = []
    num=0
    for _lambda in lambdas:
        X_hat, y_hat = build_matrix_y(1e4)
        w = np.zeros(X_hat.shape[1])
        # X_hat, y_hat, w = load_matrix_y_w("X_hat.mat", "y_hat.mat", "w.mat")
        ytX2: np.ndarray = 2 * y_hat.transpose().dot(X_hat)
        yty: np.ndarray = y_hat.transpose().dot(y_hat)
        arguments = [X_hat, ytX2, yty]
        # f_ris = f_lls(w, (XtX, ytX2, yty))
        # g_ris = grad_lls(w, (XtX, ytX2))
        tol = 1e-14
        opt_bfgs = {'gtol': 1e-16, 'disp': True}
        opt_lbfgs = {'disp': True, 'ftol': 1e-14, 'gtol': 1e-14, 'maxls': 1000, 'maxcor': 20}
        ris = minimize(fun=f_lls, x0=w, args=arguments, method='L-BFGS-B', jac=grad_lls, tol=tol, options=opt_lbfgs,
                       callback=call)

        # ris = fmin_bfgs(f_lls, w, grad_lls, arguments, tol, full_output=True, callback=call)
        # ris = fmin_l_bfgs_b(f_lls, w, grad_lls, arguments, m=50, disp=1, callback=call, factr=1e2, maxls=10000)

        w_bfgs = ris.x
        w_star, res, rnk, s = lstsq(X_hat, y_hat)
        ris_np = np.linalg.lstsq(X_hat, y_hat, rcond=None)
        norm_err = np.linalg.norm(w_star - w_bfgs)

        print("{0:1.16e}".format(norm_err))
        print("{0:1.5e}".format(np.linalg.cond(X_hat)))
        print("{0:1.16e}".format(np.linalg.norm(X_hat.dot(w_star) - y_hat)))

        error = np.zeros(len(list_xk))
        for i in range(len(list_xk)):
            error[i] = np.linalg.norm(list_xk[i] - w_star)
        np.savetxt("scipy_errors{}.csv".format(num), error, delimiter=",")
        num += 1

        scipy_errors.append(error)
        plt.figure(figsize=(10, 4), dpi=80)
        plt.title('{0} l={1} ||w-w*|| error with lambda={2:1.1e} K(A)={3:1.4e} ||w_last-w*||={4:1.3e}'
                  .format(name, opt_lbfgs['maxcor'], _lambda, cond(X_hat), norm_err))
        plt.plot(np.log10(error), label="Scipy ArmijoWolfe")
        linear = np.log10([error[0] / pow(2, i) for i in range(50)])
        quadratic = np.log10([error[0]] + [error[0] / pow(2, 2 ** i) for i in range(7)])
        plt.plot(linear, label="linear")
        plt.plot(quadratic, label="quadratic")
        plt.xlabel("iterations")
        plt.ylabel("Log error")
        plt.grid(True)
        plt.legend()
        # plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
