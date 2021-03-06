function [x, k, errors, residuals] = mgd(X,  x0, x_star, resid_fun, tol, eta, alpha, b, fast, verbose)
% Performs an Gradient descent approach with momentum.
% Inputs:
%       X           input matrix
%       x0          starting point
%       x_star      optimal solution
%       resid_fun   function which compute the residual.
%       tol         tolerance for our norm(gradient)
%       eta         learning rate hyperparameter
%       aplha       momentum hyperparameter
%       b           arrays of expected values, \hat{X}'*\hat{y}
%       verbose     print state of the SMD during the iterations
%
% Output:
%       x           solution
%       k           number of steps
%       errors      array of errors
%       residuals   array of residuals
%
% Reference:
%       Algorithm 6 from our report.
%
% Created by Niko Dalla Noce, Alessandro Ristori and Simone Rizzo
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
%eta = 0.00001; %BLS(f, df, x, 1e-4, 0.5, 1);
while(n2df>tol && k<2e4)
    
    %r_k = - df(x);
    n2df = norm(r_k); 
    %r_k_q = r_k'*r_k;
    %A_rk = X*r_k;
      
    % update parameters
    %eta = r_k_q/(A_rk'*A_rk);

    dx = eta * r_k  +  alpha * dx;
    x = x + dx;
    
    if(~fast || mod(k + 1 , 50) == 0)
         r_k = - df(x);
    else
         r_k = r_k  - eta * (X' * A_rk);
    end
    
    
    errors = [errors norm(x-x_star)];

    %residuals = [residuals resid_fun(x)];
    if (~mod(k, 50) && verbose)
    fprintf('%5d %1.16e\n', k, n2df);
    end
    k = k + 1;
end
end