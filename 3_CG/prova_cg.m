clear;

addpath ../utilities;
[X_hat, y_hat, w, w_star] = build_matrices("../datasets/ML-CUP21-TR.csv", 1e2);
[f_lls, grad_lls] = build_lls(X_hat, y_hat);
rmpath ../utilities;

% Compute the solution using conjugate 

[m,n] = size(X_hat);
A_aux = [eye(m, m), X_hat; X_hat', zeros(n, n)];
[m_aux, n_aux] = size(A_aux);
b_aux = [y_hat; zeros(m_aux-length(y_hat),1)];

A = X_hat' * X_hat;
b = X_hat' * y_hat;
x0 = randn(length(A),1);
L = chol(A, 'lower');
M = L'*L;

tol = 1e-14;
[x, k] = cg(A, x0, b, tol);
[x_w] = conjgrad(A, b, x0);
%[x_p, k_p] = pre_cg(A, x0, b, tol);
[x_m] = pcg(A, b, tol);
[x_ti, k_ti] = cg_tizio(x0, A, b, tol);

disp(norm(x - w_star))
disp(norm(x_w - w_star))
%disp(norm(x_p - w_star))
disp(norm(x_ti - w_star))
disp(norm(x_m - w_star))