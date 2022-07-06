clear;

addpath ../utilities;
[X_hat, y_hat, w, w_star] = build_matrices("../datasets/ML-CUP21-TR.csv", 1e-2);
[f_lls, grad_lls] = build_lls(X_hat, y_hat);
rmpath ../utilities;

% Compute the solution using conjugate 

[m,n] = size(X_hat);
A_aux = [eye(m, m), X_hat; X_hat', zeros(n, n)];
[m_aux, n_aux] = size(A_aux);
b_aux = [y_hat; zeros(m_aux-length(y_hat),1)];

A = X_hat' * X_hat;
b = X_hat' * y_hat;
x0 = zeros(length(A),1);
%x0 = x0/norm(x0);

tol = 1e-12;
[x, k] = cg(A, x0, b, tol);
[x_w] = conjgrad(A, b, x0);
[L, D] = ldl(A);
[x_p, k_p] = pre_cg(A, x0, b, tol, eye(size(A)));
%[x_mb] = pcg_mat(A, b, tol, 100, L*L');
[x_m] = pcg(A, b, tol, 100);
%[x_ti, k_ti] = cg_tizio(x0, A, b, tol);

disp(norm(x - w_star))
%disp(norm(x_w - w_star))
disp(norm(x_p - w_star))
%disp(norm(x_mb - w_star))
disp(norm(x_m - w_star))