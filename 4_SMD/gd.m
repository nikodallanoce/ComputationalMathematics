function [x, k] = gd(grad, x, tol, eta, alpha)
%{
Performs an Gradient descent approach with momentum.
Inputs:
    g       gradient of the function f
    x       starting point
    tol     tolerance for our norm(gradient)
    eta     learning rate hyperparameter
    aplha   momentum hyperparameter
Output: solution of the problem and number of steps.
%}

k = 0;
df = grad(x)';
v = 0;
while(norm(df)>tol && k<100)
    v = v*alpha - eta*df;          % compute the direction v
    x = x + v;                     % uptade x
    df = grad(x)';                 % compute the new gradient
    k = k + 1;
    fprintf('%5d %1.2e\n', k, norm(df));
end
end