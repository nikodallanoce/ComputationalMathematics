clear;

addpath ../utilities;
[X_hat, y_hat, ~, w_star] = build_matrices("../datasets/ML-CUP21-TR.csv", 1e4);
rmpath ../utilities;

% Compute the solution using conjugate gradient
A = X_hat' * X_hat;
b = X_hat' * y_hat;
w0 = zeros(length(A), 1);
[w_opt, k_opt, w_hist_opt] = cg_opt(sparse(X_hat), w0, b, 1e-14, 1000); % optimal version
[w, k, w_hist] = cg(A, w0, b, 1e-14, 1000); % non-optimal version

% Compute the relative errors and the final residual
rel_errors = vecnorm(w_hist - w_star) / norm(w_star);
residual = norm(X_hat * w_hist(:, end) - y_hat) / norm(y_hat);

rel_errors_opt = vecnorm(w_hist_opt - w_star) / norm(w_star);
residual_opt = norm(X_hat * w_hist_opt(:, end) - y_hat) / norm(y_hat);
