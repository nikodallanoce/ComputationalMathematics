function [X_hat, y_hat, w, w_star] = build_matrices(dataset, lambda)
% Build the matrices needed for the methods to run
% Inputs:
%       dataset     a dataset of points
%       lambda      hyper-parameter that influences the condition number
%
% Output:
%       X_hat       an m \times n matrix where the upper part is the
%                   transpose of the input dataset and the lower one
%                   is the product of lambda with an identity matrix
%       y_hat       array of expected values, where the upper part is a
%                   random array and the lower one is filled with zeros
%       w           starting point
%       w_star      optimal solution
%
% Created by Niko Dalla Noce, Alessandro Ristori and Simone Rizzo

% Build the X matrix from the dataset
X = readtable(dataset);
X = table2array(X);
X = X(:, 2:end);

% Build \hat{X} and \hat{y}
[m, n0] = size(X);
X_hat = [X'; lambda.*eye(m)];
[m, n] = size(X_hat);
y_hat = [randn(n0, 1); zeros(m-n0, 1)];

% Build our initial starting point and retrieve the optimal solution
w = randn(n, 1);
w_star = X_hat\y_hat;
end