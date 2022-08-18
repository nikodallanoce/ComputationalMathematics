function [f_lls, grad_lls] = build_lls(X_hat, y_hat)
% Build the matlab function that will be used by the methods
% Inputs:
%       X_hat       input matrix
%       y_hat       array of expected values
%
% Output:
%       f_lls       least mean squares function, (1.2) from our report
%       grad_lls    least mean squares gradient, (1.3) from our report
%
% Reference:
%       Equations (1.2) and (1.3) from our report.
%
% Created by Niko Dalla Noce, Alessandro Ristori and Simone Rizzo

% Build the LLS function and gradient
X_hat = sparse(X_hat);
y_hat = sparse(y_hat);
ytX2 = 2.*y_hat'* X_hat;
yty = y_hat'*y_hat;
Xw_f= @(w)X_hat*w;
grad_lls = @(w) 2.*Xw_f(w)'*X_hat - ytX2;
f_lls = @(w) fun(w, Xw_f, ytX2, yty);
end

function func = fun(w, Xw_f, ytX2, yty)
    Xw= Xw_f(w);
    func = Xw'*Xw - ytX2*w + yty;
end