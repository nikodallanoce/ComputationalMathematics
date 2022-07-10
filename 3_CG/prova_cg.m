clear;
%rng(1);
addpath ../utilities;
[X_hat, y_hat, w, w_star] = build_matrices("../datasets/ML-CUP21-TR.csv", 1e-4);
rmpath ../utilities;

% Compute the solution using conjugate 
%{
X_hat = randn(1500, 500);
y_hat = randn(1500,1);
w_star = X_hat\y_hat;
%}

[m,n] = size(X_hat);
A_aux = [eye(m, m), X_hat; X_hat', zeros(n, n)];
[m_aux, n_aux] = size(A_aux);
b_aux = [y_hat; zeros(m_aux-length(y_hat),1)];

A = X_hat' * X_hat;
b = X_hat' * y_hat;


x0 = zeros(length(A),1);
%x0 = x0/norm(x0);

tol = 1e-14;
[x, k, errors] = cg_opt(sparse(X_hat), x0, b, tol, w_star);
[x_z, k_z, ~] = cg(A, x0, b, tol, w_star);
[x_w] = conjgrad(A, b, x0);
alpha = max(sum(abs(A),2)./diag(A))-2;
%L = full(ichol(sparse(A), struct('type','ict','droptol',1e-10, 'diagcomp',alpha)));
%[x_p, k_p] = pre_cg(A, zeros(length(A),1), b, tol, L*L');
%[x_mb] = pcg_mat(A, b, tol, 100);
[x_m] = pcg(A, b, tol, 100);
%[x_ti, k_ti] = cg_tizio(x0, A, b, tol);

disp(norm(x - w_star))
disp(norm(X_hat * x - y_hat)/norm(y_hat))

e = @(i) power(((sqrt(cond(X_hat)) - 1)/(sqrt(cond(X_hat)) + 1)),i);
err = @(x) 0.5*((x-w_star)'*X_hat')*X_hat*(x-w_star);
err_0 = err(x0);
err_star = err(x);
e_t = err_star/err_0;
max_iter = ceil(0.5*sqrt(cond(A))*log(1/e_t));