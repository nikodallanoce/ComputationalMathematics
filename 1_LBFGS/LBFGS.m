function [x_next, k, residuals, errors, p_errors] = LBFGS(x0, f, grad, X, y, l, tol, Wolfe, verbose, x_star)
xk = x0; % current point
grad_k = grad(x0)'; % gradient at the current point
s_mem = zeros(length(xk), l); % displacements between next and current points
y_mem = zeros(length(xk), l); % displacements between next and current gradients
x_next = zeros(length(xk));
H0 = eye(size(X, 2));
residuals = norm(X*xk-y)/norm(y);
errors = norm(xk-x_star)/norm(x_star);
p_errors = abs(f(xk)-f(x_star));
for k=1:1:1000
    pk = -compute_direction(grad_k, s_mem, y_mem, H0, k); % search direction
    % compute the step size by doing a line search
    if Wolfe
        %alpha = ArmijoWolfe(f, grad, pk, xk); %OLD implementation
        alpha = strong_wolfe_line_search(f,grad,pk,xk); %JP implementation
        %alpha = Strongwolfe(f,grad,pk,xk,f(xk),grad(xk)'); %CH implementation 
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
    errors = [errors norm(xk-x_star)/norm(x_star)];
    p_errors = [p_errors abs(f(xk)-f(x_star))];

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

function r = compute_direction(gradient, s, y, H0, k)
%{
Computes the search direction, p, for the current iteration of the LBFGS
method.
Input:
    gradient: (array) gradient of the function evaluated at the current point
    s: (matrix) l displacements defined as y_k=x_{k+1}-x_k, each displacement
    is a column
    y: (matrix) l difference between gradients defined as y_k=\nabla
    f_{k+1}-\nabla f_{k}, each element is a column
    H0: (matrix) nxn identity matrix
    k: (int) current iteration
Output:
    r: (array) current search direction defined as H_k \nabla f_k
%}

q = gradient;
[~, nc] = size(s);
if k <= nc
    nc = k-1;
end
alpha = zeros(nc);
rho = zeros(nc);
for i = nc:-1:1
    rho(i) = 1/(y(:, i)'*s(:, i));
    alpha(i) = rho(i).* s(:, i)' * q;
    q = q - alpha(i).* y(:, i);
end

gamma = 1;
if k > 1
    gamma = s(:, nc)'*y(:, nc) / norm(y(:, nc))^2;
end

H0 = gamma * H0;
r = H0 * q;

for i = 1:nc
    beta = rho(i) * y(:, i)' * r;
    r = r + s(:, i)*(alpha(i) - beta);
end
end
