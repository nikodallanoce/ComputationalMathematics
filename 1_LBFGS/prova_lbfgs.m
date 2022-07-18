clear;
addpath ../utilities;
[X_hat, y_hat, w, w_star] = build_matrices("../datasets/ML-CUP21-TR.csv", 1e4);
[f_lls, grad_lls] = build_lls(X_hat, y_hat);
rmpath ../utilities;

% Compute the solution using L-BFGS
w = zeros(size(w));
b = X_hat'*y_hat;
[w_our, k, errors] = LBFGS_risto(w, sparse(X_hat), b, 20, 1e-14, true, w_star);

disp(norm(w_our-w_star)/norm(w_star));
lin = zeros(size(errors));
quad = zeros(size(errors));
lin(1) = errors(1);
quad(1) = errors(1);
for i = 2:length(errors)
    lin(i) = lin(1)/power(2, i);

    if(i<10)
        quad(i) = quad(1)/power(2, power(2, i));
    end
end

semilogy(lin);
hold on
semilogy(quad);
semilogy(errors)
hold off