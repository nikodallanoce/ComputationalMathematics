clear;

%rng(1);

addpath ../utilities;
[X_hat, y_hat] = build_matrices("../datasets/ML-CUP21-TR.csv", 1e2);
rmpath ../utilities;

% Compute the thin QR factorization
[Q1y, R1] = thinqr(X_hat, y_hat);

% Compute the solution
x = R1 \ Q1y;

% Compute matlab solution
[Q_t, R_t] = qr(X_hat, "econ");
x_star = X_hat\y_hat;

[Q1, R1] = thinqr(X_hat);
%upper = cond(X_hat)*norm(Q1'*(X_hat*x-y_hat))/norm(X_hat*x)
