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
lambda = [1, 1e-2, 1e-4, 1e-8];
runs = 10;
[residues, errors, times, iters, config] = run_configurations_lbfgs(runs, l, lambda, 1e-8, dataset, y, false);

[r, ~] = size(residues);
stats = strings(r, 5);

for i=1 : r
    row = {config(i),...
           sprintf("res: %e +- %e", mean(residues(i,:)), std(residues(i,:))),...
           sprintf("err: %e +- %e", mean(errors(i,:)), std(errors(i,:))),...
           sprintf("time: %e +- %e", mean(times(i,:)), std(times(i,:))),...
           sprintf("it: %e +- %e", mean(iters(i,:)), std(iters(i,:)))};
    stats(i,:) = row;
end
