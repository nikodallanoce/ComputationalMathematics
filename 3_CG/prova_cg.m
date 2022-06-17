clear;

addpath ../utilities;
[X_hat, y_hat, w, w_star] = build_matrices("../datasets/ML-CUP21-TR.csv", 1e0);
[f_lls, grad_lls] = build_lls(X_hat, y_hat);
rmpath ../utilities;

% Compute the solution using conjugate 

A = X_hat' * X_hat;
b = X_hat' * y_hat;
x0 = randn(length(A),1);
[x, k] = cg(A, x0, b, 1e-14);
[x_m] = pcg(A, b, 1e-14);
[x_i, k_i] = cg_tizio(x0, A, b, 1e-14);