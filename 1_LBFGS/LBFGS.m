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

function [alpha] = strong_wolfe(func, grad, x0, f0, g0, p)
% function [alpha] = strong_wolfe(func,x0,f0,g0,p)
% Compute a line search to satisfy the strong Wolfe conditions.
% Algorithm 3.5. Page 60. "Numerical Optimization". Nocedal & Wright.
% INPUTS:
%  func: objective function handle.
%  x0: [n,1] initial design vector.
%  f0: initial function evaluation.
%  g0: [n,1] initial objective gradient vector.
%  p: [n,1] search direction vector.
% OUTPUTS:
% alpha: search length

% initialize variables
c1 = 1e-4;
c2 = 0.2;
alpha_max = 3;
alpha_im1 = 0;
alpha_i = 1;
f_im1 = f0;
dphi0 = transpose(g0)*p;
i = 0;
max_iters = 10;

% search for alpha that satisfies strong-Wolfe conditions
while true
  
  x = x0 + alpha_i*p;
  f_i = func(x);
  g_i = grad(x)';
  if (f_i > f0 + c1*dphi0) || ( (i > 1) && (f_i >= f_im1) )
    alpha = alpha_zoom(func, grad, x0, f0,g0,p,alpha_im1,alpha_i);
    break;
  end
  dphi = transpose(g_i)*p;
  if ( abs(dphi) <= -c2*dphi0 )
    alpha = alpha_i;
    break;
  end
  if ( dphi >= 0 )
    alpha = alpha_zoom(func, grad, x0,f0,g0,p,alpha_i,alpha_im1);
    break;
  end
  
  % update
  alpha_im1 = alpha_i;
  f_im1 = f_i;
  alpha_i = alpha_i + 0.5*(alpha_max-alpha_i);
  
  if (i > max_iters)
    alpha = alpha_i;
    break;
  end
  
  i = i+1;
  
end

end

function [alpha] = alpha_zoom(func, grad, x0,f0,g0,p,alpha_lo,alpha_hi)
% function [alpha] = alpha_zoom(func,x0,f0,g0,p,alpha_lo,alpha_hi)
% Algorithm 3.6, Page 61. "Numerical Optimization". Nocedal & Wright.
% INPUTS:
%  func: objective function handle.
%  x0: [n,1] initial design vector.
%  f0: initial objective value.
%  g0: [n,1] initial objective gradient vector.
%  p: [n,1] search direction vector.
%  alpha_lo: low water mark for alpha.
%  alpha_hi: high water mark for alpha.
% OUTPUTS:
%  alpha: zoomed in alpha.

% initialize variables
c1 = 1e-4;
c2 = 0.2;
i = 0;
max_iters = 15;
dphi0 = transpose(g0)*p;

while true
  alpha_i = 0.5*(alpha_lo + alpha_hi);
  alpha = alpha_i;
  x = x0 + alpha_i*p;
  f_i = func(x);
  g_i = grad(x)';
  x_lo = x0 + alpha_lo*p;
  f_lo = func(x_lo);
  if ( (f_i > f0 + c1*alpha_i*dphi0) || ( f_i >= f_lo) )
    alpha_hi = alpha_i;
  else
    dphi = transpose(g_i)*p;
    if ( ( abs(dphi) <= -c2*dphi0 ) )
      alpha = alpha_i;
      break;
    end
    if ( dphi * (alpha_hi-alpha_lo) >= 0 )
      alpha_hi = alpha_lo;
    end
    alpha_lo = alpha_i;
  end
  i = i+1;
  if (i > max_iters)
    alpha = alpha_i;
    break;
  end
end

end
