clear;
format long e;
addpath ../utilities;

[X_hat, y_hat, w, w_star] = build_matrices("../datasets/ML-CUP21-TR.csv", 1e2);
[f_lls, grad_lls] = build_lls(X_hat, y_hat);
rmpath ../utilities;

% Compute the solution using standard momentum descent (heavy ball)
b = X_hat' * y_hat;
x0 = zeros(length(w_star),1);
tol = 9e-13;

[rows_number, ~] = size(X_hat);

resid_fun = @(xk) norm(X_hat*xk - y_hat)/norm(y_hat);

momentums = [0, 5e-4, 5e-2, 0.1, 0.3, 0.5];
labels = string(size(momentums));
for i=1:length(momentums)
    mom = momentums(i);
    [x, ~, error_mom, ~] = mgd_eqn(X_hat, x0, w_star, resid_fun, tol, mom, b, 1e4, false, true);
    labels(i) = num2str(mom);
    errors = error_mom./norm(w_star);
    disp(norm(x-w_star)/norm(w_star));
    semilogy(errors, 'LineWidth', 1);
    if i==1
        hold on;
        grid on;
    end
end
hold off;
legend(labels);
title("Comparing error curves using of the same problem, varying momentum", "Interpreter", "latex");
xlabel("steps");
ylabel("$\frac{||w - w^{*}||}{ ||w^{*}||}$", 'Interpreter','latex');
%{
[x, ~, error_zero, ~] = mgd_eqn(X_hat, x0, w_star, resid_fun, tol, 0.0, b, 1e4, false, true);
disp(norm(x-w_star)/norm(w_star));

semilogy(error_zero, 'LineWidth', 1)
hold on;
semilogy(error_mom, 'LineWidth', 1);
hold off;
grid on;
s = sprintf("momentum = %.1e", mom);
legend(["no momentum", s]);
%}