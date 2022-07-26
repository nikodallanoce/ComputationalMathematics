function [xk, k, errors, residual] = LBFGS(x0, X, y, y_hat, l, tol, verbose, x_star)
% Limited memory BFGS (L-BFGS)
% Inputs:
%       x0          starting point
%       X           input matrix
%       y           array of expected values
%       l           memory size
%       tol         tolerance
%       verbose     print state of the L-BFGS during the iterations
%       x_star      optimal solution
%
% Output:
%       xk          solution
%       k           number of iterations spent by the method
%       errors      array of errors computed at each iteration
%
% Reference:
%       Algorithm 2 from our report, which is in turn based
%       on Algorithm 7.5 from Jorge Nocedal and Stephen Wright,
%       "Numerical optimization," Springer Science & Business Media, 2006.
%
% Created by Niko Dalla Noce, Alessandro Ristori and Simone Rizzo

xk = x0; % starting point
grad = @(w) X'*(X*w) - y; % gradient
grad_k = grad(x0);
sm = []; % array of displacements
ym = []; % array of differences between gradients
residual = 0;
% metrics
errors = norm(xk-x_star)/norm(x_star);

% print starting state of L-BFGS
if verbose
    fprintf('%5d %1.2e %1.2e\n', 0, errors(end), norm(grad_k));
end

for k=1:1000
    % compute search direction
    pk = -compute_direction(grad_k, sm, ym, k);

    % compute step size by exact line search
    A_pk = X*pk;
    alpha = -(grad_k'*pk)/(A_pk'*A_pk);

    % update step and gradient
    xk_prev = xk;
    grad_prev = grad_k;
    xk = xk + alpha.* pk;
    grad_k = grad(xk);

    % memory handling
    sm = [sm xk - xk_prev];
    ym = [ym grad_k - grad_prev];

    if k>l
        sm(:, 1) = [];
        ym(:, 1) = [];
    end

    if (sm(:, end)'*ym(:, end)<=0)
        fprintf("Curvature condition does not hold\n");
    end

    % compute metrics
    errors = [errors norm(xk-x_star)/norm(x_star)];

    % print current state of L-BFGS
    if verbose && (mod(k, 5) == 0 || k == 1)
         fprintf('%5d %1.2e %1.2e\n', k, errors(end), norm(grad_k));
    end

    % stop if the norm of the direction is less than the tolerance
    if norm(pk) <= tol
        residual = norm(X*xk-y_hat)/norm(y_hat);
        break;
    end
end

% print final state of L-BFGS
if verbose && mod(k, 5) ~= 0
     fprintf('%5d %1.2e %1.2e\n', k, errors(end), norm(grad_k));
end
end

function r = compute_direction(grad, s, y, k)
% Two loop recursion needed to compute the search direction for L-BFGS
% Inputs:
%       grad        gradient computed at current point
%       s           displacements
%       y           differences between gradients
%       k           current iteration
%
% Output:
%       r           H_k*\nabla f
%
% Reference:
%       Algorithm 1 from our report, which is in turn based
%       on Algorithm 7.4 from Jorge Nocedal and Stephen Wright,
%       "Numerical optimization," Springer Science & Business Media, 2006.
%
%Created by Niko Dalla Noce, Alessandro Ristori and Simone Rizzo

if k == 1
    r = grad;
else
    q = grad;
    for i = size(s, 2):-1:1
        rho(i) = 1 / (y(:, i)' * s(:, i));
        alpha(i) = rho(i).* s(:, i)' * q;
        q = q - alpha(i).* y(:, i);
    end
    
    % we do not need to explicitly use H_k^0
    gamma = s(:, end)'* y(:, end) / norm(y(:, end))^2;
    r = gamma.* q;
    
    for i = 1:size(s, 2)
        beta = rho(i) * y(:, i)' * r;
        r = r + s(:, i) * (alpha(i) - beta);
    end
end
end