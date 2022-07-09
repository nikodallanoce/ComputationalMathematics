function [x, k, errors, residuals] = gd(grad, x, x_star, resid_fun, tol, eta, alpha, n)
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

k = 0;
df = grad(x)';
v = 0;
errors = norm(x-x_star);
residuals = resid_fun(x);
while(norm(df)>tol && k<1000)
    v = v*alpha - eta*df./n;          % compute the direction v
    x = x + v;                        % uptade x
    df = grad(x)';                    % compute the new gradient
    k = k + 1;
    errors = [errors norm(x-x_star)];
    residuals = [residuals resid_fun(x)];
    fprintf('%5d %1.2e\n', k, norm(df));
end
end