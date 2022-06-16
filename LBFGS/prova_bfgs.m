clear;

% Build the X matrix from the dataset
dataset = readtable("../datasets/ML-CUP21-TR.csv");
dataset = table2array(dataset);
dataset = dataset(:, 2:end);
X = dataset;

% Build \hat{X} and \hat{y}
[m, n0] = size(X);
lambda = 1e-4;
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
[w_our, k, residuals, errors, p_errors] = LBFGS(w, f_lls, grad_lls, X_hat, y, 50, 1e-12, true, true, matlab_w);
p = zeros(1, k-1);
for i=1:1:k-1
    if p_errors(i) == 0
       p_errors(i) = 1e-16;
    end
    p(1, i) = log(p_errors(i+1))./log(p_errors(i));
end