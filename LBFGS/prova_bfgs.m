clear;
rng(1);

% Build the X matrix from the dataset
full_path = 'C:/Users/Simone/Documents/ComputationalMathematics/datasets';
dataset = readtable(strcat(full_path,'/ML-CUP21-TR.csv'));
dataset = table2array(dataset);
X = dataset(:, 2:end);

% Build \hat_{X} and \hat_{y}
[m, n0] = size(X);
lambda = 1e-5;
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
[ouptut, k] = LBFGS(w, f_lls, X, grad_lls, 14, 1e-5);

% Compute LFBGS for different configurations
l = [3, 5, 10];
tol = [1e-2, 1e-4, 1e-6];
lambda = [1e-2, 1e-3, 1e-4, 1e-5];
run_configurations(l, tol, lambda, f_lls, grad_lls, w, dataset);