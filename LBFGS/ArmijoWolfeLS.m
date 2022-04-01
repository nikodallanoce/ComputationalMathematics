function a = ArmijoWolfeLS(f, grad, alpha, m1, m2, tau, a, pk)
%{
Armijo-Wolfe conditions for finding the optimal stepsize.
Input:  
      f            function to minimize
      grad         derivative of the function f
      alpha        function with one parameter, alpha(a)= xk + a*pk;
      m1           hyperparameter of armijo condition, it must be between
                   0 < m1 < 1.
      m2           hyperparameter of Wolf condition, it must be 0<m1<m2<1.
      tau          parameter for decreasing the stepsize a.
      a            stepsize initial value
      pk           direction vector
Output:
      a            stepsize
%}
xk = alpha(0);
phi_zero = f(xk);
grad_zero = grad(xk);
while(all((f(alpha(a))> phi_zero + m1*a*grad_zero)) && (-grad(alpha(a))*pk > -m2*grad_zero*pk))
    a=a*tau;
end
end