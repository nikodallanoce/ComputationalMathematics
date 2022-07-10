function [x_k, k, errors, errors_A] = cg_opt(A, x_0, b, tol, w_star)
r_k = A'*(A*x_0) - b; % residual
p_k = -r_k; % search direction
x_k = x_0; % current point
k = 0; % iteration
tolb = tol*norm(b); % stop condition
norm_A = @(w) (w-w_star)'*A'*A*(w-w_star);
errors = norm(w_star-x_k);
errors_A = norm_A(x_0); 


while(norm(r_k)>tolb && k<1000)
    [x_k, r_k, p_k] = iteration(A, r_k, p_k, x_k);
    errors = [errors norm(w_star - x_k)];
    errors_A = [errors_A norm_A(x_k)];
    k = k+1;
end
end

function [x_k_next, r_k_next, p_k_next] = iteration(A, r_k, p_k, x_k)
% computations to save time efficiently
r_k_q = r_k'*r_k;
A_pk = A*p_k;

% update parameters
a_k = r_k_q/(A_pk'*A_pk);
x_k_next = x_k + a_k * p_k;
r_k_next = r_k + A' * a_k * A_pk;
B_k_next = (r_k_next' * r_k_next) / r_k_q;
p_k_next = -r_k_next + B_k_next*p_k;
end