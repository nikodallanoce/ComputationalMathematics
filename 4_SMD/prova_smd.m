clear;
format long e;
addpath ../utilities;
rng(1);
[X_hat, y_hat, w, w_star] = build_matrices("../datasets/ML-CUP21-TR.csv", 1e-4);
[f_lls, grad_lls] = build_lls(X_hat, y_hat);
rmpath ../utilities;

% Compute the solution using standard momentum descent (heavy ball)
b = X_hat' * y_hat;
x0 = zeros(length(w_star),1);
tol = 1e-12;

[rows_number, ~] = size(X_hat);

resid_fun = @(xk) norm(X_hat*xk - y_hat)/norm(y_hat);
%grad_lls = @(x) (X_hat'*(X_hat*x) - X_hat'*y_hat)';
grad_lls = @(r, eta, dfX) r - eta*dfX ;

[x, k, errors, residuals] = mgd(f_lls, X_hat, grad_lls, x0, w_star, resid_fun, tol, 1e-6, 0.1, rows_number, b);
disp(norm(x-w_star)/norm(w_star));

lin = zeros(size(errors));
sub = zeros(size(errors));
lin(1) = errors(1);
sub(1) = errors(1);
for i = 2:length(errors)
    lin(i) = lin(1)/sqrt(i);
    sub(i) = sub(1)/i;
end

semilogy(lin);
hold on
semilogy(sub);
semilogy(errors)
hold off