clear;
addpath ../utilities;
[X_hat, y_hat, w, w_star] = build_matrices("../datasets/ML-CUP21-TR.csv", 1e4);
[f_lls, grad_lls] = build_lls(X_hat, y_hat);
rmpath ../utilities;

% Compute the solution using L-BFGS
w = zeros(size(w)); % starting point
b = X_hat' * y_hat;
[w_our, k, w_hist] = LBFGS(w, sparse(X_hat), b, 20, 1e-14, true, 1000);

% Compute the relative errors and the final residual
rel_errors = vecnorm(w_hist - w_star) / norm(w_star);
residual = norm(X_hat * w_hist(:, end) - y_hat) / norm(y_hat);