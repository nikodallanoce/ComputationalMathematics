clear;
addpath ../utilities;

rng(1);
% l_list = 5:20;
lambda_list = [1e4, 1e2, 1, 1e-2, 1e-4];
% lambda = 1e-4;
verbose = false;
[a,lenght_l] = size(lambda_list);
tol = 1e-12;

errors = -ones(lenght_l, 1000);
for i=1: lenght_l
    [X_hat, y_hat, w, w_star] = build_matrices("../datasets/ML-CUP21-TR.csv", lambda_list(i));
    [f_lls, grad_lls] = build_lls(X_hat, y_hat);
    w = zeros(size(w));
    b = X_hat'*y_hat;
    [~, k, error, resid] = LBFGS_risto(w, sparse(X_hat),b, y_hat, 5, tol, verbose, w_star);
    [a,lenght_e] = size(error);
    errors(i, 1:lenght_e) = error;
    semilogy(error);
    if i==1
        hold on;
        grid on;
        xlabel ('steps');
        ylabel('$\frac{||w - w^{*}||}{ ||w^{*}||}$', Interpreter='latex');
        title('Errors by varing Lambda, l=5')
    end    
end
legend(string(lambda_list));