clear;
format long e;
rng(1);
% Build the X matrix from the dataset
dataset = readtable('ML-CUP21-TR.csv');
dataset = table2array(dataset);
dataset = dataset(:, 2:end);
X = dataset;

% Build \hat_{X} and \hat_{y}
[m0, n0] = size(X);
lambda = 1e-2;
X = [X'; lambda.*eye(m0)];
[m, n] = size(X);
y = [randn(n0, 1); zeros(m-n0, 1)];

Wolfe = true;
w = randn(n, 1);
lambdas = [1, 1e-2, 1e-4, 1e-8];
L = [5, 10, 15, 20];

for l=L
    res = -ones(1000,length(lambdas));
    err = -ones(1000,length(lambdas));
    for i=1: length(lambdas)
        curr_l = lambdas(i);
        X = [dataset'; curr_l.*eye(m0)];
        [~, n] = size(X);
        XtX = X'*X;
        ytX2 = 2.*y'*X;
        yty = y'*y;
        grad_lls = @(x) 2.*x'*XtX - ytX2;
        f_lls = @(x) x'*XtX*x - ytX2*x + yty;
    
        [~, k, residues, errors] = LBFGS(w, f_lls, X, grad_lls, l, 1e-8, Wolfe, y);
        res(1:k+1,i)=residues;
        err(1:k+1,i)=errors;
    end
    save_s=sprintf("results/conf_lambda_l_%d_res_wolfe.csv", l);
    csvwrite(save_s, res);
    save_s=sprintf("results/conf_lambda_l_%d_err_wolfe.csv", l);
    csvwrite(save_s, err);
end

