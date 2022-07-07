
clear;
%rng(1);
addpath ../utilities;
addpath ArmijoWolfeImplementations\;
[X_hat, y_hat, w, w_star] = build_matrices("../datasets/ML-CUP21-TR.csv", 1e-4);
[f_lls, grad_lls] = build_lls(X_hat, y_hat);
rmpath ../utilities;

% Compute the solution using L-BFGS
w = zeros(size(w));
[w_our, k, residuals, errors, p_errors] = LBFGS(w, f_lls, grad_lls, X_hat, y_hat, 20, 1e-14, true, true, w_star);

[m,n] = size(errors);
quad = [errors(1,1)];
lin = zeros(size(errors));
lin(1) = errors(1,1);
for i = 1:n
    lin(i) = lin(1)/(2^i);
    if (i<6)
        quad = [quad quad(1)/power(2, 2^i)];
    end
end

disp(errors(1,end))

semilogy(linspace(1, length(errors), length(errors)), errors);
hold on;
grid on;
semilogy(linspace(1, length(quad), length(quad)), quad);
semilogy(linspace(1, length(lin), length(lin)), lin);
legend(["BFGS", "quadratic", "linear"]);
hold off;