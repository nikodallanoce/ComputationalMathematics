clear;

% Build the X matrix from the dataset
dataset = readtable("../datasets/ML-CUP21-TR.csv");
dataset = table2array(dataset);
dataset = dataset(:, 2:end);
X = dataset;

% Build \hat{X} and \hat{y}
[m, n0] = size(X);
lambda = 1e-8;
X_hat = [X'; lambda.*eye(m)];
[m, n] = size(X_hat);
y = [randn(n0, 1); zeros(m-n0, 1)]; % This is actually \hat{y}

% Build our initial starting point and retrieve optimal solution
w = randn(n, 1);
matlab_w = X_hat\y;

% Build the LLS function and gradient
XtX = X_hat' * X_hat;
ytX2 = 2.*y'* X_hat;
yty = y'*y;
grad_lls = @(w) 2.*w'*XtX - ytX2;
f_lls = @(w) w'*XtX*w - ytX2*w + yty;

% Compute the solution using L-BFGS
[w_our, k, residue, error] = LBFGS(w, f_lls, X_hat, grad_lls, 5, 10e-14, false, y);

% Compute LFBGS for different configurations
[residues, errors, times, iters, config] = run_configurations_lbfgs(10, l, lambda, 1e-8, dataset, y, false);
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