clear;
rng(1);

% Build the X matrix from the dataset
dataset = readtable('ML-CUP21-TR.csv');
dataset = table2array(dataset);
X = dataset(:, 2:end);

% Build \hat_{X} and \hat_{y}
[m, n0] = size(X);
X = [X'; eye(m)];
[m, n] = size(X);
y = [randn(n0, 1); zeros(m-n0, 1)];

w = randn(n, 1);

grad_lls = @(x) 2.*x'*(X'*X) - 2.*y'*X;
f_lls = @(x) x'*(X'*X)*x - 2.*y'*X*x + y'*y;

l = 19;

ris = X\y;

[w, k] = LBFGS(w, f_lls, X, grad_lls, l, 1e-8);