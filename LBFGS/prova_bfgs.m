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

XtX=(X'*X);
ytX2= 2.*y'*X;
yty=y'*y;

grad_lls = @(x) 2.*x'*XtX - ytX2;
f_lls = @(x) x'*XtX*x - ytX2*x + yty;

l = 19;

%ris = X\y;

[w, k] = LBFGS(w, f_lls, X, grad_lls, l, 1e-8);