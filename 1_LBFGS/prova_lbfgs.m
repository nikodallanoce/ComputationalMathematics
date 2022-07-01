clear;
rng(1);
addpath ../utilities;
addpath ArmijoWolfeImplementations\;
[X_hat, y_hat, w, w_star] = build_matrices("../datasets/ML-CUP21-TR.csv", 1e-4);
[f_lls, grad_lls] = build_lls(X_hat, y_hat);
rmpath ../utilities;

% Compute the solution using L-BFGS
[w_our, k, residuals, errors, p_errors] = LBFGS(w, f_lls, grad_lls, X_hat, y_hat, 10, 1e-12, true, true, w_star);
rmpath ArmijoWolfeImplementations\;

p = zeros(1, k-1);
for i=1:1:k-1
    if p_errors(i+1) < 1e-14
       p_errors(i+1) = 1e-14;
    end
    p(1, i) = log(p_errors(i+1))./log(p_errors(i));
end

semilogy(linspace(1, length(errors), length(errors)), errors)