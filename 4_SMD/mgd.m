function [x_best, k, errors, residuals] = mgd(f, X, grad, x0, x_star, resid_fun, tol, eta, alpha, n, b)
%{
Performs an Gradient descent approach with momentum.
Inputs:
    g         gradient of the function f
    x         starting point
    x_star    matlab solution
    resid_fun function which compute the residual.
    tol       tolerance for our norm(gradient)
    eta       learning rate hyperparameter
    aplha     momentum hyperparameter
    n         dimension of the matrix, used for normalization
              of the gradient.
Output: solution of the problem, number of steps, errors and residuals.
%}
%addpath("../1_LBFGS\ArmijoWolfeImplementations");
k = 0;
x=x0;
errors = norm(x-x_star);
residuals = resid_fun(x);
%eta = strong_wolfe_line_search(f, grad, -df, x);
df = @(x) X'*(X*x)- b;
r_k = -df(x);
n2df = inf;
patience = 0;
patience_zig = 0;
dx = 0;
min_grad = inf;
x_best = x;
%eta = 0.00001; %BLS(f, df, x, 1e-4, 0.5, 1);
while(n2df>tol && k<2e4)
    
    r_k = - df(x);
    n2df = norm(r_k); 
    r_k_q = r_k'*r_k;
    A_rk = X*r_k;
      
    % update parameters
    eta = r_k_q/(A_rk'*A_rk);
    dx = eta * r_k  +  alpha * dx;
    x = x + dx;
    
    %{
    if(mod(k + 1 , 50) == 0)
         r_k = - df(x);
    else
         r_k = r_k  + (X' * A_rk);
         r_k = r_k;
    end
    %}
    
    errors = [errors norm(x-x_star)];

    %residuals = [residuals resid_fun(x)];
    if (~mod(k, 50))
    fprintf('%5d %1.16e\n', k, n2df);
    end
    k = k + 1;
    x_best = x;
end
end

function alpha = BLS(f, grad_xk, xk, c1, tau, alpha)
f_xk = f(xk);
iter = 0;
while (f(xk - alpha * grad_xk) > f_xk - c1 * alpha * (grad_xk' * grad_xk))
    alpha = alpha * tau;
    iter = iter +1;
end
%disp(iter);
end