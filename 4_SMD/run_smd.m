clear; clear global;
format long e;
addpath ../utilities;

[X_hat, y_hat, w, w_star] = build_matrices("../datasets/ML-CUP21-TR.csv", 1e0);
[f_lls, grad_lls] = build_lls(X_hat, y_hat);


% Compute the solution using standard momentum descent (heavy ball)
b = X_hat' * y_hat;
x0 = zeros(length(w_star),1);
tol = 1e-12;

grad = @(x) X_hat'*(X_hat*x)- b;
max_iters = 1e4;

%global rel_errors, global residual;
global metrics;
metrics.rel_errors = zeros(max_iters, 1);
metrics.residual = zeros(max_iters, 1);

[x, k, w_hist] = smd(X_hat, grad, x0, tol, 0.03, max_iters, true,...
    @(x,k) callback(x, k, w_star, X_hat, y_hat));
rmpath ../utilities;
disp(norm(x-w_star)/norm(w_star));
%rel_errors = vecnorm(w_hist - w_star) / norm(w_star);
%residual = norm(X_hat * w_hist(:, end) - y_hat) / norm(y_hat);
