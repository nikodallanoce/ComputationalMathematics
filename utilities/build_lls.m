function [f_lls, grad_lls] = build_lls(X_hat, y_hat)
% Build the LLS function and gradient
XtX = X_hat' * X_hat;
ytX2 = 2.*y_hat'* X_hat;
yty = y_hat'*y_hat;
grad_lls = @(w) 2.*w'*XtX - ytX2;
f_lls = @(w) w'*XtX*w - ytX2*w + yty;
end