function a = BLS(f, grad, alpha, m1, tau, a)
al = alpha(0);
phi_zero = f(al);
grad_zero = grad(al);
while(f(alpha(a))> phi_zero + m1*a*grad_zero)
    a=a*tau;
end
end