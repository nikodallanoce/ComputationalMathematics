function [x, k, errors] = mgd_eqn(X, grad, x0, x_star, tol, alpha, max_iters, fast, verbose)
% Performs an Gradient descent approach with momentum.
% Inputs:
%       X           input matrix
%       grad        gradient of the function
%       x0          starting point
%       x_star      optimal solution
%       tol         tolerance for our norm(gradient)
%       eta         learning rate hyperparameter
%       aplha       momentum hyperparameter
%       max_iters   maximum number of iterations
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

k = 0;
x=x0;
errors = norm(x-x_star);
r_k = -grad(x);
n2df = inf;
dx = 0;
while(n2df>tol && k<max_iters)
    
    %r_k = - df(x);
    r_k_q = r_k'*r_k;
    A_rk = X*r_k;
    n2df = norm(r_k);

    % Update parameters
    eta = r_k_q/(A_rk'*A_rk);

    dx = eta * r_k  +  alpha * dx;
    x = x + dx;
    
    if(~fast || mod(k + 1 , 50) == 0)
         r_k = - grad(x);
    else
         r_k = r_k  - eta * (X' * A_rk);
    end
    
    errors = [errors norm(x-x_star)];

    if (~mod(k, 50)&&verbose)
        fprintf('%5d %1.16e\n', k, n2df);
    end
    k = k + 1;
end
end
