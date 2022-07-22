function [xk, k, errors, residual] = LBFGS_risto(x0, X, y, y_hat, l, tol, verbose, x_star)
xk = x0;
grad = @(w) X'*(X*w) - y;
grad_k = grad(x0);
sm = [];
ym = [];
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
        fprintf("Curvature condition does not hold");
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
if k == 1
    r = grad;
else
    q = grad;
    for i = size(s, 2):-1:1
        rho(i) = 1 / (y(:, i)' * s(:, i));
        alpha(i) = rho(i).* s(:, i)' * q;
        q = q - alpha(i).* y(:, i);
    end
    
    gamma = s(:, end)'* y(:, end) / norm(y(:, end))^2;
    r = gamma.* q;
    
    for i = 1:size(s, 2)
        beta = rho(i) * y(:, i)' * r;
        r = r + s(:, i) * (alpha(i) - beta);
    end
end
end