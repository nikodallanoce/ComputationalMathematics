clear;
format long e;
addpath ../utilities;
[X_hat, y_hat, w, w_star] = build_matrices("../datasets/ML-CUP21-TR.csv", 1e-4);
[f_lls, grad_lls] = build_lls(X_hat, y_hat);
rmpath ../utilities;

% Compute the solution using standard momentum descent (heavy ball)
b = X_hat' * y_hat;
x0 = zeros(length(w_star),1);
tol = 1e-12;

[rows_number, ~] = size(X_hat);

resid_fun = @(xk) norm(X_hat*xk-y_hat)/norm(y_hat);
[x, k, errors, residuals] = mgd(f_lls, grad_lls, x0, w_star, resid_fun, tol,0.001, 0.9, rows_number, 1);
disp(norm(x-w_star));