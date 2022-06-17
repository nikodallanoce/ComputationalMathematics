function [x_next, k, residuals, errors] = LBFGS(x0, f, grad, X, y, l, tol, Wolfe, verbose, x_star)
xk = x0; % current point
grad_k = grad(x0)'; % gradient at the current point
s_mem = zeros(length(xk), l); % displacements between next and current points
y_mem = zeros(length(xk), l); % displacements between next and current gradients
x_next = zeros(length(xk));
H0 = eye(size(X, 2));
residuals = norm(X*xk-y)/norm(y);
errors = norm(xk-x_star);
for k=1:1:2000
    pk = -compute_direction(grad_k, s_mem, y_mem, H0, k); % search direction
    % compute the step size by doing a line search
    if Wolfe
        alpha = ArmijoWolfe(f, grad, pk, xk);
    else
        alpha = BLS(f, grad, xk, pk, 1e-4, 0.5, 1);
    end
    
    % compute the next point, gradient
    x_next = xk + alpha.*pk;
    grad_next = grad(x_next)';

    % compute the displacements
    x_displacement = x_next-xk;
    yk = grad_next - grad_k;

    % memory handling
    if k > l
        s_mem(:, 1:end-1) = s_mem(:, 2:end);
        y_mem(:, 1:end-1) = y_mem(:, 2:end);
        s_mem(:, end) = x_displacement;
        y_mem(:, end) = yk;
    else
        s_mem(:, k) = x_displacement;
        y_mem(:, k) = yk;
    end
    
    % update the parameters
    grad_k = grad_next;
    xk = x_next;

    % print current state of L-BFGS
    if verbose && (mod(k, 5) == 0 || k == 1)
         fprintf('%5d %1.2e %1.2e\n', k, alpha, norm(grad_k));
    end

    % compute metrics
    residuals = [residuals norm(X*xk-y)/norm(y)];
    errors = [errors norm(xk-x_star)];

    % stop if the gradient is smaller than the tolerance
    if Wolfe && norm(grad_k) < tol
        break;
    else
        if k <= l
            norm_y = norm(y_mem(k));
        else
            norm_y = norm(y_mem(end));
        end
        if norm_y < tol || norm(grad_k) < tol
            break;
        end
    end
end
if verbose && mod(k, 5) ~= 0
    fprintf('%5d %1.2e %1.2e\n', k, alpha, norm(grad_k));
end
end