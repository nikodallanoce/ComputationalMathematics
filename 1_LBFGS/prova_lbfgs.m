clear;
addpath ../utilities;
addpath ArmijoWolfeImplementations\;
[X_hat, y_hat, w, w_star] = build_matrices("../datasets/ML-CUP21-TR.csv", 1e-4);
[f_lls, grad_lls] = build_lls(X_hat, y_hat);
rmpath ../utilities;
% Compute the solution using L-BFGS
w = zeros(size(w));
[w_our, k, residuals, errors, p_errors] = LBFGS(w, f_lls, grad_lls, X_hat, y_hat, 20, 1e-10, true, true, w_star);
%[w_our, k, residuals, errors] = LBFGS(w, f_lls, grad_lls, X_hat, y_hat, 10, 1e-14, true, w_star);
rmpath ArmijoWolfeImplementations\;

%{
p = zeros(1, k-1);
for i=1:1:k-1
    if p_errors(i+1) < 1e-14
       p_errors(i+1) = 1e-14;
%     end
    p(1, i) = log(p_errors(i+1))./log(p_errors(i));
end
%}

%save('X_hat.mat', 'X_hat')
%save('y_hat.mat', 'y_hat')
%save('w.mat', 'w')

% semilogy(linspace(1, length(errors), length(errors)), errors)
disp(norm(w_our-w_star)/norm(w_star));
lin = zeros(size(errors));
quad = zeros(size(errors));
lin(1) = errors(1);
quad(1) = errors(1);
for i = 2:length(errors)
    lin(i) = lin(1)/power(2, i);
    quad(i) = quad(1)/power(2, power(2, i));
end

semilogy(lin);
hold on
semilogy(quad);
semilogy(errors)
hold off