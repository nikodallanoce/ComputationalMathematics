function [x_k, k, x_hist] = LBFGS(x0, X, y, l, tol, verbose, max_iters, callback)
% Limited memory BFGS (L-BFGS)
% [x_k, k, x_hist] = LBFGS(x0, X, y, l, tol, verbose, max_iters[, callback])
%
% If you do not want the history of all the computed x points,
% use the following alternative version.
% [x_k, k] = LBFGS(x0, X, y, l, tol, verbose, max_iters[, callback])
%
% Inputs:
%       x0          starting point
%       X           input matrix
%       y           array of expected values of the form \hat{X}'*\hat{y}
%       l           memory size
%       tol         tolerance
%       verbose     print state of the L-BFGS during the iterations
%       max_iters   maximum number of iterations
%
% Output:
%       xk          solution
%       k           number of iterations spent by the method
%       x_hist      array that keeps track of all the computed points
%
% Reference:
%       Algorithm 2 from our report, which is in turn based
%       on Algorithm 7.5 from Jorge Nocedal and Stephen Wright,
%       "Numerical optimization," Springer Science & Business Media, 2006.
%
% Created by Niko Dalla Noce, Alessandro Ristori and Simone Rizzo

x_k = x0; % starting point
grad = @(w) X'*(X*w) - y; % gradient
grad_k = grad(x0);
sm = zeros(length(x_k), l); % matrix of displacements
ym = zeros(length(x_k), l); % matrix of differences between gradients

% keep track of all the points computed by the method
if(nargout>2)
    x_hist = zeros(length(x_k), max_iters+1);
    x_hist(:, 1) = x_k;
end

if exist("callback", "var")
    callback(x_k, 1);
end

% print starting state of L-BFGS
if verbose
    fprintf("\tk norm(grad)\n");
    fprintf('%5d %1.2e\n', 0, norm(grad_k));
end

nc = 1; % number of filled columns of sm and ym
for k=1:max_iters
    % compute search direction
    pk = -compute_direction(grad_k, sm, ym, nc);

    % compute step size by exact line search
    A_pk = X*pk;
    alpha = -(grad_k'*pk)/(A_pk'*A_pk);

    % update step and gradient
    xk_prev = x_k;
    grad_prev = grad_k;
    x_k = x_k + alpha.* pk;
    grad_k = grad(x_k);

    % memory handling
    if k > l
        sm(:, 1:end-1) = sm(:, 2:end);
        ym(:, 1:end-1) = ym(:, 2:end);
        sm(:, end) = x_k - xk_prev;
        ym(:, end) = grad_k - grad_prev;
        nc = l;
    else
        sm(:, k) = x_k - xk_prev;
        ym(:, k) = grad_k - grad_prev;
        nc = k;
    end

    if (sm(:, nc)' * ym(:, nc) <= 0)
        fprintf("Curvature condition does not hold\n");
    end

    % save current point
    if (nargout>2)
        x_hist(:, k+1) = x_k;
    end

    if exist("callback", "var")
        callback(x_k, k+1);
    end

    % print current state of L-BFGS
    if verbose && (mod(k, 5) == 0 || k == 1)
         fprintf('%5d %1.2e\n', k, norm(grad_k));
    end

    % stop if the norm of the direction is less than the tolerance
    if norm(pk) <= tol
        break;
    end
end

% print final state of L-BFGS
if verbose && mod(k, 5) ~= 0
     fprintf('%5d %1.2e\n', k, norm(grad_k));
end
if (nargout > 2)
    x_hist = x_hist(:, 1:k+1);
end
end

function r = compute_direction(grad, s, y, nc)
% Two loop recursion needed to compute the search direction for L-BFGS
% Inputs:
%       grad        gradient computed at current point
%       s           displacements
%       y           differences between gradients
%       nc          number of filled columns of s and y
%
% Output:
%       r           H_k*\nabla f
%
% Reference:
%       Algorithm 1 from our report, which is in turn based
%       on Algorithm 7.4 from Jorge Nocedal and Stephen Wright,
%       "Numerical optimization," Springer Science & Business Media, 2006.
%
% Created by Niko Dalla Noce, Alessandro Ristori and Simone Rizzo

if nc == 1
    r = grad;
else
    q = grad;
    for i = nc:-1:1
        rho(i) = 1 / (y(:, i)' * s(:, i));
        alpha(i) = rho(i).* s(:, i)' * q;
        q = q - alpha(i).* y(:, i);
    end
    
    % we do not need to explicitly use H_k^0
    gamma = s(:, nc)'* y(:, nc) / norm(y(:, nc))^2;
    r = gamma.* q;
    
    for i = 1:nc
        beta = rho(i) * y(:, i)' * r;
        r = r + s(:, i) * (alpha(i) - beta);
    end
end
end