function [x_k, k, x_hist] = cg_opt(A, x_0, b, tol, max_iters, callback)
% Conjugate gradient (cg) optimal version in which we do not explicitly
% compute the matrix X_hat'*X_hat
% [x_k, k, x_hist] = cg_opt(A, x_0, b, tol, max_iters)
% Inputs:
%       A           sparse input matrix
%       x0          starting point
%       b           array of expected values of the form \hat{X}'*\hat{y}
%       tol         tolerance
%       max_iters   maximum number of iterations
%
% Output:
%       x_k         solution
%       k           number of iterations
%       x_hist      array that keeps track of all the computed points
%
% Reference:
%       Algorithm 5 from our report.
%
% Created by Niko Dalla Noce, Alessandro Ristori and Simone Rizzo

r_k = A'*(A*x_0) - b; % residual
p_k = -r_k; % search direction
x_k = x_0; % current point
tolb = tol * norm(b); % stop condition

% keep track of all the points computed by the method
if (nargout>2)
    x_hist = zeros(length(x_k), max_iters+1);
    x_hist(:, 1) = x_k;
end

if exist("callback", "var")
    callback(x_k, 1);
end

for k=1:max_iters
    [x_k, r_k, p_k] = iteration(A, r_k, p_k, x_k);

    if exist("callback", "var")
       callback(x_k, k+1);
    end
    if (nargout>2)
        x_hist(:, k+1) = x_k;
    end
    if norm(r_k) <= tolb
        break;
    end
end
if (nargout>2)
    x_hist = x_hist(:, 1:k+1);
end
end

function [x_k_next, r_k_next, p_k_next] = iteration(A, r_k, p_k, x_k)
% Single iteration of conjugate gradient (cg)
% Inputs:
%       A           input matrix
%       r_k         residual at the k-th iteration
%       p_k         search direction at the k-th iteration
%       x_k         current point
%
% Output:
%       x_k_next    next point
%       r_k_next    next residual
%       p_k_next    next search direction
%
% Reference:
%       Algorithm 5 from our report.
%
% Created by Niko Dalla Noce, Alessandro Ristori and Simone Rizzo

% computations to save time efficiently
r_k_q = r_k'*r_k;
A_pk = A*p_k;

% update parameters
a_k = r_k_q / (A_pk'*A_pk);
x_k_next = x_k + a_k * p_k;
r_k_next = r_k + A' * a_k * A_pk;
B_k_next = (r_k_next' * r_k_next) / r_k_q;
p_k_next = -r_k_next + B_k_next*p_k;
end