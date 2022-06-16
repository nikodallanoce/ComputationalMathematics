function [X_hat, y_hat, w, w_star] = build_matrices(dataset, lambda)
% Build the X matrix from the dataset
X = readtable(dataset);
X = table2array(X);
X = X(:, 2:end);

% Build \hat{X} and \hat{y}
[m, n0] = size(X);
X_hat = [X'; lambda.*eye(m)];
[m, n] = size(X_hat);
y_hat = [randn(n0, 1); zeros(m-n0, 1)];

% Build our initial starting point and retrieve optimal solution
w = randn(n, 1);
w_star = X_hat\y_hat;
end