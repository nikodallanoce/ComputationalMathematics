
clear;
%rng(1);
addpath ../utilities;

lambdas = [1e4, 1e2, 1e0, 1e-2, 1e-4];
runs = 5;

times = zeros(length(lambdas), runs);
errors = -1 * ones(length(lambdas), runs, 100);
ks = zeros(length(lambdas), runs);
tol = 1e-14;
for r=1:runs
    for i=1:length(lambdas)
        lambda = lambdas(i);
        [X_hat, y_hat, w, w_star] = build_matrices("../datasets/ML-CUP21-TR.csv", lambda);
        b = X_hat' * y_hat;
        x0 = zeros(length(w_star), 1);
        time_start = tic;
        [x_k, k, err] = cg_opt(sparse(X_hat), x0, b, tol, w_star);
        times(i,r) = toc(time_start);
        ks(i, r) = k;
        errors(i, r, 1: length(err)) = err;
    end
    disp(r);
end
