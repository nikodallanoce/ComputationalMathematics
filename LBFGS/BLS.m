function a = BLS(f, grad, alpha, m1, tau, a, xk)
al = alpha(0);
phi_zero = f(al);
grad_zero = grad(xk);
while(f(alpha(a))> phi_zero + m1*a*grad_zero)
    a=a*tau;
end
end