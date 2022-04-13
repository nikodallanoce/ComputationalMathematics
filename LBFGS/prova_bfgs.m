clear;
%rng(1);

% Build the X matrix from the dataset
dataset = readtable("../datasets/ML-CUP21-TR.csv");
dataset = table2array(dataset);
dataset = dataset(:, 2:end);
X = dataset;

% Build \hat_{X} and \hat_{y}
[m, n0] = size(X);
lambda = 1e-2;
X = [X'; lambda.*eye(m)];
[m, n] = size(X);
y = [randn(n0, 1); zeros(m-n0, 1)];

%{
w = randn(n, 1);

% Build the function and its gradient
XtX = X'*X;
ytX2 = 2.*y'*X;
yty = y'*y;

grad_lls = @(x) 2.*x'*XtX - ytX2;
f_lls = @(x) x'*XtX*x - ytX2*x + yty;

% Matlab output
ris = X\y;

% Compute LBFGS

[output_bls, k_bls] = LBFGS(w, f_lls, X, grad_lls, 14, 1e-8, false);
[output, k] = LBFGS(w, f_lls, X, grad_lls, 14, 1e-8, true);

fprintf("norm_res_bls= %.8e | norm_res_wolfe=%.8e | norm_res_matl=%.8e\n \n", norm(X*output_bls-y), norm(X*output-y), norm(X*ris-y))
fprintf("diff: %.8e\n \n", norm(output-output_bls));
fprintf("rel_err -> BLS: %.15e | Wolfe: %.15e \n \n", norm(output_bls-ris)/norm(ris), norm(output-ris)/norm(ris))
%}
%{
[output, k, residuals, errors] = LBFGS(w, f_lls, X, grad_lls, 14, 1e-8, true, y);
[output_bls, k_bls, residuals_bls, errors_bls] = LBFGS(w, f_lls, X, grad_lls, 14, 1e-8, false, y);
disp(norm(output-output_bls));
disp(norm(X*output-y)/norm(y));
disp(norm(X*output_bls-y)/norm(y));
%}

% Compute LFBGS for different configurations
l = [5, 10, 15, 20];
lambda = [1, 1e-2, 1e-4, 1e-9];
run_configurations_lbfgs(10, l, lambda, 1e-8, dataset, y, true);