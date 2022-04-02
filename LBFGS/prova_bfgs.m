clear;
% rng(1);

% Build the X matrix from the dataset
dataset = readtable("../datasets/ML-CUP21-TR.csv");
dataset = table2array(dataset);
X = dataset(:, 2:end);

% Build \hat_{X} and \hat_{y}
[m, n0] = size(X);
lambda = 1e-1;
X = [X'; lambda.*eye(m)];
[m, n] = size(X);
y = [randn(n0, 1); zeros(m-n0, 1)];

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
%{
[output, k] = LBFGS(w, f_lls, X, grad_lls, 14, 1e-8, true);
[output_bls, k_bls] = LBFGS(w, f_lls, X, grad_lls, 14, 1e-8, false);
disp(norm(output-output_bls));
disp(norm(X*output-y));
disp(norm(X*output_bls-y));
disp(norm(X*ris-y));
%}

% Compute LFBGS for different configurations
l = [3, 5, 10, 16];
lambda = [1, 1e-2, 1e-4, 1e-9];
run_configurations(l, lambda, f_lls, grad_lls, y, w, dataset);