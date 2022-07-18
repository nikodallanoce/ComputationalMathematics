clear;
addpath ../utilities;
addpath ArmijoWolfeImplementations\;
[X_hat, y_hat, w, w_star] = build_matrices("../datasets/ML-CUP21-TR.csv", 1e-4);
[f_lls, grad_lls] = build_lls(X_hat, y_hat);
rmpath ../utilities;
% Compute the solution using L-BFGS
w = zeros(size(w));
y_hat = X_hat'*y_hat;
%[w_our, k, residuals, errors, p_errors] = LBFGS(w, f_lls, grad_lls, X_hat, y_hat, 20, 1e-12, true, true, w_star);
%[w_our, k, residuals, errors] = LBFGS_risto(w, f_lls, grad_lls, X_hat, y_hat, 30, 1e-12, true, w_star);
[w_our, k, errors] = LBFGS(w, f_lls, grad_lls, sparse(X_hat), y_hat, 20, 1e-14, true, w_star);
rmpath ArmijoWolfeImplementations\;


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