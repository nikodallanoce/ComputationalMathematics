
clear;
addpath ../utilities;

lambdas = [1e4, 1e2, 1e0, 1e-2, 1e-4];
runs = 10;
max_iter = 30;
times = -ones(runs, length(lambdas));
errors = -ones(runs, length(lambdas), max_iter);
errors_A = -ones(runs, length(lambdas), max_iter);
residuals = -ones(runs, length(lambdas));
ks = -ones(runs, length(lambdas));
tol = 1e-14;

for r=1:runs
    for i=1:length(lambdas)
        lambda = lambdas(i);
        [X_hat, y_hat, w, w_star] = build_matrices("../datasets/ML-CUP21-TR.csv", lambda);
        b = X_hat' * y_hat;
        x0 = zeros(length(w_star), 1);
        X_hat = sparse(X_hat);
        time_start = tic;
        [x, k, err, err_A] = cg_opt(X_hat, x0, b, tol, w_star);
        times(r,i) = toc(time_start);
        residuals(r,i) = norm(X_hat * x - y_hat)/norm(y_hat);
        ks(r,i) = k;
        n2w = norm(w_star);
        err = err(end);
        err_A = [err_A(1),err_A(end)];
        errors(r,i, 1:length(err)) = err/n2w;
        errors_A(r,i, 1:length(err_A)) = err_A;
    end
    disp(r);
end
%semilogy(cell2mat(errors(1,1)))