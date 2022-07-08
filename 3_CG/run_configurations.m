
clear;
addpath ../utilities;

lambdas = [1e4, 1e2, 1e0, 1e-2, 1e-4];
runs = 5;

times = zeros(length(lambdas), runs);
errors = cell(length(lambdas), runs);
residuals = zeros(length(lambdas), runs);
ks = zeros(length(lambdas), runs);
tol = 1e-14;

for r=1:runs
    for i=1:length(lambdas)
        lambda = lambdas(i);
        [X_hat, y_hat, w, w_star] = build_matrices("../datasets/ML-CUP21-TR.csv", lambda);
        b = X_hat' * y_hat;
        x0 = zeros(length(w_star), 1);
        X_hat = sparse(X_hat);
        time_start = tic;
        [x, k, err] = cg_opt(X_hat, x0, b, tol, w_star);
        times(i,r) = toc(time_start);
        residuals(i,r) = norm(X_hat * x - y_hat)/norm(y_hat);
        ks(i, r) = k;
        n2w = norm(w_star);
        errors(i, r) = mat2cell(err/n2w, 1, length(err));
    end
    disp(r);
end
%semilogy(cell2mat(errors(1,1)))