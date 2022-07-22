function [x_next, k, errors, residual] = LBFGS(x0, f, grad, X, y, l, tol, verbose, x_star)
xk = x0;                      % current point
grad = @(x) X'*(X*x) - y;
grad_k = grad(x0);           % gradient at the current point
s_mem = zeros(length(xk), l); % displacements between next and current points
y_mem = zeros(length(xk), l); % displacements between next and current gradients
x_next = zeros(length(xk));

errors = norm(xk-x_star)/norm(x_star);
residual = 0;
k = 1;
while(k<1000)

    pk = -compute_direction(grad_k, s_mem, y_mem, k); % search direction
    
    % compute the step size by doing a line search
    A_pk = X*pk;
    alpha = -(grad_k'*pk)/(A_pk'*A_pk);
    %alpha = strong_wolfe(f, grad, xk, f(xk), grad_k, pk);

    % compute the next point, gradient
    x_next = xk + alpha.*pk;
    grad_next = grad(x_next);

    % compute the displacements
    x_displacement = x_next-xk;
    yk = grad_next - grad_k;

    if(x_displacement'*yk <= 0)
        warning("curvature");
    end

    grad_k = grad_next;
    xk = x_next;
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

    % print current state of L-BFGS
    if verbose && (mod(k, 5) == 0 || k == 1)
         fprintf('%5d %1.2e %1.2e\n', k, alpha, norm(grad_k));
    end

    % compute metrics     
    errors = [errors norm(xk-x_star)/norm(x_star)];

    % stop if the gradient is smaller than the tolerance
    k = k+1;
    if norm(pk) < tol %|| norm(yk) < tol
        residual = norm(X*xk-y)/norm(y);
        break;
    end
   
    
end
if verbose && mod(k, 5) ~= 0
    fprintf('%5d %1.2e %1.2e\n', k, alpha, norm(grad_k));
end
end

function r = compute_direction(gradient, s, y, k)
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
    gamma = s(:, nc)'*y(:, nc) / (y(:, nc)'*y(:, nc));
end

r = gamma * q;

for i = 1:nc
    beta = rho(i) * y(:, i)' * r;
    r = r + s(:, i)*(alpha(i) - beta);
end
end