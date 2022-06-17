clear;

% Build the X matrix from the dataset
dataset = readtable("../datasets/ML-CUP21-TR.csv");
dataset = table2array(dataset);
D = dataset(:, 2:end);

% Build \hat{X} and \hat{y}
[m, n0] = size(D);
lambda = 1e-8;
X_hat = [D'; lambda.*eye(m)];
[m, n] = size(X_hat);
y = [randn(n0, 1); zeros(m-n0, 1)]; % This is actually \hat{y}

lambda = [1e4, 1e2, 1, 1e-2, 1e-4];
l = [5, 10, 15, 20];

[residues, errors, times, iters, config] = run_configurations_lbfgs(1, l, lambda, 1e-12, D, y, true);

y_axis = reshape(errors(20,1,:), [1000,1]);
y_axis = y_axis(y_axis ~=-1);
x_axis = linspace(1, length(y_axis), length(y_axis));
semilogy(x_axis, y_axis);