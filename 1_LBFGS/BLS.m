function alpha = BLS(f, grad, xk, p, c1, tau, alpha)
f_xk = f(xk);
grad_xk = grad(xk);
while f(xk + alpha * p) > f_xk + c1 * alpha * grad_xk * p
    alpha = alpha * tau;
end
end