function [x, k, errors, residuals, final_err] = run(lambda, eta, alpha, verbose)
    format long e;
    addpath ../utilities;
    [X_hat, y_hat, w, w_star] = build_matrices("../datasets/ML-CUP21-TR.csv", lambda);
    [f_lls, grad_lls] = build_lls(X_hat, y_hat);
    rmpath ../utilities;
    
    % Compute the solution using standard momentum descent (heavy ball)
    b = X_hat' * y_hat;
    x0 = zeros(length(w_star),1);
    tol = 1e-12;
    
    [rows_number, ~] = size(X_hat);
    
    resid_fun = @(xk) norm(X_hat*xk-y_hat)/norm(y_hat);
    [x, k, errors, residuals] = mgd(f_lls, grad_lls, x0, w_star, resid_fun, tol, eta, alpha, rows_number, verbose);
    final_err = norm(x-w_star);
end