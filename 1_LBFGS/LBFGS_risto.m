function [xk, k, residuals, errors] = LBFGS_risto(x0, f, grad, X, y, l, tol, verbose, x_star)
xk = x0;
grad_k = grad(x0)';
sm = [];
ym = [];

% metrics
residuals = norm(X*xk-y)/norm(y);
errors = norm(xk-x_star)/norm(x_star);

if verbose
    fprintf('%5d %1.2e %1.2e\n', 0, errors(end), norm(grad_k));
end

for k=1:1000
    pk = -compute_direction(grad_k, sm, ym, k);
    if pk' * grad_k > 0
        %pk = -pk;
    end

    alpha = strong_wolfe_line_search(f, grad, pk, xk); % step size
    %alpha = BLS(f, grad, xk, pk, 1e-4, 0.5, 1);
    %alpha = ArmijoWolfe(f, grad, pk, xk);

    xk_prev = xk;
    grad_prev = grad_k;
    xk = xk + alpha.* pk;
    grad_k = grad(xk)';

    sm = [sm xk - xk_prev];
    ym = [ym grad_k - grad_prev];

    if k>l
        sm(:, 1) = [];
        ym(:, 1) = [];
    end

    if (sm(end)'*ym(end)<=0)
        warning("Curvature condition does not hold");
        %break;
    end

    % compute metrics
    residuals = [residuals norm(X*xk-y)/norm(y)];
    errors = [errors norm(xk-x_star)/norm(x_star)];

    % print current state of L-BFGS
    if verbose && (mod(k, 5) == 0 || k == 1)
         fprintf('%5d %1.2e %1.2e\n', k, errors(end), norm(grad_k));
    end

    if norm(grad_k) <= tol || norm(ym(end)) <= tol
        break;
    end
end
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
    r = 1e-4.*q;
    
    for i = 1:size(s, 2)
        beta = rho(i) * y(:, i)' * r;
        r = r + s(:, i) * (alpha(i) - beta);
    end
end
end