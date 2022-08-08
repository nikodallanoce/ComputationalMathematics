clear;

addpath ../utilities;
[X_hat, y_hat] = build_matrices("../datasets/ML-CUP21-TR.csv", 1e4);
rmpath ../utilities;

% Compute the thin QR factorization
[Q1y, R1] = thinqr(X_hat, y_hat);

% Compute our solution
opts.UT = true;
w = linsolve(R1, Q1y, opts);

% Compute matlab solution
[Q_t, R_t] = qr(X_hat, "econ");
w_star = X_hat\y_hat;

rel_error = norm(w - w_star) / norm(w_star);
residual = norm(X_hat * w - y_hat) / norm(y_hat);
