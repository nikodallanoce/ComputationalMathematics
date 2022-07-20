
clear;
addpath ../utilities;

lambdas = [1e4, 1e2, 1e0, 1e-2, 1e-4];
runs = 4;
max_iter = 1e4;
times = -ones(runs, length(lambdas));
errors = -ones(runs, length(lambdas));
errors_A = -ones(runs, length(lambdas), max_iter);
residuals = -ones(runs, length(lambdas));
ks = -ones(runs, length(lambdas));
tol = 7e-13;
beta = 0.0;
for r=1:runs
    for i=1:length(lambdas)
        if(i>1)
            beta = 0.05;
        end
        lambda = lambdas(i);
        [X_hat, y_hat, w, w_star] = build_matrices("../datasets/ML-CUP21-TR.csv", lambda);
        b = X_hat' * y_hat;
        x0 = zeros(length(w_star), 1);
        X_hat = sparse(X_hat);
        time_start = tic;
        resid_fun = @(xk) norm(X_hat*xk - y_hat)/norm(y_hat);
        [x, k, err, ~] = mgd_eqn(X_hat, x0, w_star, resid_fun, tol, beta, b, max_iter, false, false);
        %[x,FLAG,RELRES,k] = lsqr(X_hat, y_hat, tol, 100);
        times(r,i) = toc(time_start);
        residuals(r,i) = norm(X_hat * x - y_hat)/norm(y_hat);
        ks(r,i) = k;
        n2w = norm(w_star);
        err = err(end);
        %err_A = [err_A(1),err_A(end)];
        errors(r,i) = err/n2w;
        %errors_A(r,i, 1:length(err_A)) = err_A;
    end
    disp(r);
end