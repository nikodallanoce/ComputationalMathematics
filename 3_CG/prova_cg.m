clear;

addpath ../utilities;
[X_hat, y_hat, w, w_star] = build_matrices("../datasets/ML-CUP21-TR.csv", 1e-2);
[f_lls, grad_lls] = build_lls(X_hat, y_hat);
rmpath ../utilities;

% Compute the solution using conjugate 

A = X_hat' * X_hat;
b = X_hat' * y_hat;
x0 = randn(length(A),1);
L = chol(A, 'lower');
M = L'*L;

tol = 1e-14;
[x, k] = cg(A, x0, b, tol);
[x_p, k_p] = pre_cg(A, M, x0, b, tol);
[x_m] = pcg(A, b, tol);
[x_i, k_i] = cg_tizio(x0, A, b, tol);