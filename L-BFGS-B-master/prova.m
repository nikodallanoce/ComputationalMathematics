clear;
addpath ../utilities;
addpath ArmijoWolfeImplementations\;
[X_hat, y_hat, w, w_star] = build_matrices("../datasets/ML-CUP21-TR.csv", 1e0);
[f_lls, grad_lls] = build_lls(X_hat, y_hat);
rmpath ../utilities;

% Compute the solution using L-BFGS
w = zeros(size(w));

% solve the unconstrained rosenbrock problem
l = -inf*ones(size(w));
u = -l;
opts = struct('display', true, 'xhistory', true, 'max_iters', 500, 'tol', 1e-12);
grad_lls = @(x) grad_lls(x)';
[x, xhistory] = LBFGSB(f_lls, grad_lls, w , l , u ,opts, w_star);
semilogy(xhistory);         