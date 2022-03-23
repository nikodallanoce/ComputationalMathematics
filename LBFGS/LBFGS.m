function [x_next, k] = LBFGS(x0, f, X, grad, l, tol)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
xk=x0;
k=0;
grad_k = grad(x0)';
len = size(X,2);
s = [];
y = []; %ones(length(grad_k),1);
x_next=zeros(length(xk));

while(norm(grad_k)>tol)
    pk = -compute_direction(grad_k, s, y, len, k);
    alpha = BLS(f, grad, @(alpha)xk + alpha.*pk, 1e-4, 0.8, 1, xk);
    x_next = xk + alpha.*pk;
    yk = grad(x_next)' - grad_k;

    if k > l
        s = [s(:,2:end) x_next-xk];
        y = [y(:, 2:end) yk];
    else
        s= [s x_next-xk];
        y = [y yk];
    end
    grad_k=grad(x_next)';
    xk=x_next;
    k=k+1;
end
end