clear;
%rng(1);
addpath ../utilities;
[X_hat, y_hat, w, w_star] = build_matrices("../datasets/ML-CUP21-TR.csv", 1e0);
[f_lls, grad_lls] = build_lls(X_hat, y_hat);
rmpath ../utilities;

% Compute the solution using L-BFGS
[w_our, k, residuals, errors, p_errors] = LBFGS(w, f_lls, grad_lls, X_hat, y_hat, 20, 1e-10, true, true, w_star);

%save('X_hat.mat', 'X_hat')
%save('y_hat.mat', 'y_hat')
%save('w.mat', 'w')

[m,n] = size(errors);
quad = zeros(size(errors));
lin = zeros(size(errors));
lin(1) = errors(1,1);
quad(1) = errors(1,1);
for i = 2:n
    lin(i) = lin(1)/(2^i);
    if (i<10)
        quad(i) = quad(1)/power(2, 2^i);
    end
end


semilogy(linspace(1, length(errors), length(errors)), errors);
hold on;
grid on;
semilogy(linspace(1, length(quad), length(quad)), quad);
semilogy(linspace(1, length(lin), length(lin)), lin);
legend(["BFGS", "quadratic", "linear"]);
hold off;