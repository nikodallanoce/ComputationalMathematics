function [f_lls, grad_lls] = build_lls(X_hat, y_hat)
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