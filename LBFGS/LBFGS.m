function [x_next, k] = LBFGS(x0, f, X, grad, l, tol)
%{
LBFGS computes the solution to the linear least squares problem following
algorithm 7.4 by Jorge Nocedal and Stephen J Wright. Numerical optimization.
Input:
    x0: (array) starting point
    f: (function) function to minimize
    X: (matrix) mxn matrix
    grad: (function) gradient of the function
    l: (int) number of steps to keep in memory
    tol: (float) tollerance on the error
Ouptut:
    x_next: (array) solution, or close to it, to the minimization problem
    k: (int) steps taken
%}

xk=x0;
k=0; % number of iterations
grad_k = grad(x0)';
len = size(X, 2);
s = [];
y = [];
x_next=zeros(length(xk));
norm_y= 9999;
I = eye(len);
while(norm(grad_k)>tol && norm_y>tol)
    if mod(k, 50)==0
        disp(k)
    end
    pk = -compute_direction(grad_k, s, y, I, k); % search direction
    alpha = BLS(f, grad, @(alpha)xk + alpha.*pk, 1e-4, 0.5, 1, xk);
    x_next = xk + alpha.*pk;
    grad_next = grad(x_next)';
    yk = grad_next - grad_k;

    % if k exceeds the available memory, remove the oldest s and y

    [s,y] = memory_handling(s, y, xk, x_next, yk, k, l);

    % update the parameters
    grad_k = grad_next;
    xk=x_next;
    k=k+1;

    % if the last gradient didn't have a great change from the previous
    norm_y = norm(y(end));
end
end