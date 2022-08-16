function [x_k, k, x_hist] = mgd_eqn(X, grad, x0, tol, alpha, max_iters, verbose)
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

x_k = x0;
% keep track of all the points computed by the method
if (nargout > 2)
    x_hist = zeros(length(x_k), max_iters+1);
    x_hist(:, 1) = x_k;
end

r_k = -grad(x_k);
delta_x = 0;

for k=1:max_iters

    n2df = norm(r_k); % computes norm 2 of the direction

    % Update parameters
    r_k_q = r_k'*r_k;
    A_rk = X*r_k;
    eta = r_k_q/(A_rk'*A_rk);
    delta_x = eta * r_k  +  alpha * delta_x;
    x_k = x_k + delta_x;
    
    r_k = - grad(x_k);
    
    if (verbose && ~mod(k, 50)), fprintf('%5d %1.16e\n', k, n2df); end
    
    if(nargout > 2), x_hist(:, k+1) = x_k; end
    if(n2df<=tol), break, end

end
if(nargout > 2)
    x_hist = x_hist(:, 1:k+1);
end
end
