clear;

lambdas = [1e4, 1e2, 1, 1e-2, 1e-4];
etas = linspace(1e-5, 1e-2);

addpath ../utilities;
[X_hat, y_hat, w, w_star] = build_matrices("../datasets/ML-CUP21-TR.csv", 1e-2);
[f_lls, grad_lls] = build_lls(X_hat, y_hat);
rmpath ../utilities;

% Compute the solution using standard momentum descent (heavy ball)
b = X_hat' * y_hat;
x0 = zeros(length(w_star),1);
tol = 1e-12;

[rows_number, ~] = size(X_hat);

resid_fun = @(xk) norm(X_hat*xk-y_hat)/norm(y_hat);
[x, k, errors, residuals] = mgd(grad_lls, x0, w_star,resid_fun, tol, 1e-3, 0.2, rows_number);
disp(norm(x-w_star));