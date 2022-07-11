function [x, k, errors, residuals] = mgd(f, grad, x, x_star, resid_fun, tol, eta, alpha, n)
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
addpath("../1_LBFGS\ArmijoWolfeImplementations");
k = 0;
v = 0;
errors = norm(x-x_star);
residuals = resid_fun(x);
%eta = strong_wolfe_line_search(f, grad, -df, x);
df = grad(x)';

while(norm(df)>tol && k<2)
    df = grad(x)';
    eta = BLS(f, grad, x, 1e-4, 0.5, 1);
    v = v*alpha - eta*df./n;          % compute the direction v
    x = x + v;                        % uptade x
    %df = grad(x)'; % compute the new gradient
    %eta = strong_wolfe_line_search(f, grad, v, x);
    k = k + 1;
    errors = [errors norm(x-x_star)];
    residuals = [residuals resid_fun(x)];
    fprintf('%5d %1.2e\n', k, norm(df));
end
end

function alpha = BLS(f, grad, xk, c1, tau, alpha)
f_xk = f(xk);
grad_xk = grad(xk);
iter = 0;
while (f(xk - alpha * grad_xk) > f_xk - c1 * alpha * (grad_xk' * grad_xk))
    alpha = alpha * tau;
    iter = iter +1;
end
disp(iter);
end