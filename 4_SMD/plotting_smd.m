clear;
addpath ../utilities;
%lambdas = [1e5, 1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5];
lambdas = [1e4, 1e2, 1e0, 1e-2, 1e-4];
alphas = [0, 0.00, 0.03, 0.04, 0.03];
times = zeros(length(lambdas),1);
errors = {};
tol = 9e-13;
ks = zeros(length(lambdas),1);
for i=1:length(lambdas)
    [X_hat, y_hat, w, w_star] = build_matrices("../datasets/ML-CUP21-TR.csv", lambdas(i));
  
    % compute the solution using conjugate gradient
    b = X_hat' * y_hat;
    resid_fun = @(xk) norm(X_hat*xk - y_hat)/norm(y_hat);
    x0 = zeros(length(w_star), 1); 
    alpha = alphas(i);
    time_elapsed = tic;
    [x, k, err, res] = mgd_eqn(X_hat, x0, w_star, resid_fun, tol, alpha, b, 1e4, false, true);
    time_elapsed = toc(time_elapsed);
    times(i) = time_elapsed;
    errors = [errors err/norm(w_star)];
    ks(i) = k;
end
rmpath ../utilities;
% build rates
linear = zeros(length(max(ks)), 1);
max_err = 0;
for i=1:length(errors)
    curr_errors = cell2mat(errors(i));
    if max(curr_errors)>max_err
        max_err = max(curr_errors);
    end
end

[m,n] = size(errors);
err1 = cell2mat(errors(1,1));
err1 = err1(1);
sub_lin = zeros(size(errors));
sub_lin(1) = err1;
for i = 2:max(ks)
    sub_lin(i) = sub_lin(1)/(i);
end

% plot errors
for i=1:length(ks)
    if (i == 1)
        semilogy(sub_lin, 'LineWidth', 1);
        hold on;   
    end
    semilogy(cell2mat(errors(i)), 'LineWidth', 1);
end
hold off;
grid on;

% insert labels
labels = strings(length(lambdas), 1);
%labels(1) = "linear";
for i=1:length(lambdas)
    labels(i) = num2str(lambdas(i), "%.1e");
end

legend(['sublinear'; labels], "Location", "southeast");
title("GD convergence speed by varying lambda values")
xlabel("steps");
ylabel("$\frac{||w - w^{*}||}{ ||w^{*}||}$", 'Interpreter','latex');
