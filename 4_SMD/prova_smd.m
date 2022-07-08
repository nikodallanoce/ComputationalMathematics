clear;

addpath ../utilities;
[X_hat, y_hat, w, w_star] = build_matrices("../datasets/ML-CUP21-TR.csv", 1e2);
[f_lls, grad_lls] = build_lls(X_hat, y_hat);
rmpath ../utilities;

% Compute the solution using standard momentum descent (heavy ball)

b = X_hat' * y_hat;
hess = 2* (X_hat' * X_hat); 
x0 = zeros(length(w_star),1);
tol = 1e-10;

[x, k] = gd(f_lls, hess, x0, b, tol, 0.5, 0.5);

disp(norm(x-w_star));