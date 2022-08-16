clear;
format long e;
addpath ../utilities;

[X_hat, y_hat, w, w_star] = build_matrices("../datasets/ML-CUP21-TR.csv", 1e0);
[f_lls, grad_lls] = build_lls(X_hat, y_hat);
rmpath ../utilities;

% Compute the solution using standard momentum descent (heavy ball)
b = X_hat' * y_hat;
x0 = zeros(length(w_star),1);
tol = 1e-12;

[rows_number, ~] = size(X_hat);

resid_fun = @(xk) norm(X_hat*xk - y_hat)/norm(y_hat);
%grad_lls = @(x) (X_hat'*(X_hat*x) - X_hat'*y_hat)';
%grad_lls = @(r, eta, dfX) r - eta*dfX ;

kappa = cond(X_hat);
beta = (sqrt(kappa) - 1)/(sqrt(kappa) + 1);
[x, k, error_abs, residuals] = mgd_eqn(X_hat, x0, w_star, resid_fun, tol, beta^2, b, 1e4, false, true);
disp(norm(x-w_star)/norm(w_star));

errors = error_abs./norm(w_star);
lin = zeros(size(errors));
sub = zeros(size(errors));
lin(1) = errors(1);
sub(1) = errors(1);
for i = 2:length(errors)
    lin(i) = lin(1)/power(2,i);
    sub(i) = sub(1)/i;
end

%semilogy(lin);
%hold on
%semilogy(sub);
semilogy(errors)
hold on;

hold off