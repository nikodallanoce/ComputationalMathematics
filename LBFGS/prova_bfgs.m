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

% Compute LFBGS for different configurations
l = [5, 10, 15, 20];
lambda = [1, 1e-2, 1e-4, 1e-9];
run_configurations_lbfgs(10, l, lambda, 1e-8, dataset, y, true);