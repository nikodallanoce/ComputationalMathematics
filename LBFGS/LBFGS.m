function [x_next, k] = LBFGS(x0, f, X, grad, l, tol, Wolfe)
%{
LBFGS computes the solution to the linear least squares problem following
algorithm 7.4 by Jorge Nocedal and Stephen J Wright. Numerical optimization.
Input:
    x0: (array) starting point
    f: (function) function to minimize
    X: (matrix) mxn matrix
    grad: (function) gradient of the function
    l: (int) number of steps to keep in memory
    tol: (float) tolerance on the error
    Wolfe: (bool) if true performs an Armijo-Wolfe line search otherwise it
    will perform a backtracking line search
Output:
    x_next: (array) solution, or close to it, to the minimization problem
    k: (int) steps taken
%}

xk=x0;
k=0; % number of iterations
grad_k = grad(x0)';
s = [];
y = [];
x_next=zeros(length(xk));
norm_y= 9999;
I = eye(size(X, 2));
while(norm(grad_k)>tol && norm_y>tol)
    pk = -compute_direction(grad_k, s, y, I, k); % search direction
    if Wolfe
        alpha = ArmijoWolfeLS(f, grad, @(alpha)xk + alpha.*pk, 1e-4, 0.9, 0.5, 1, pk);
    else
        alpha = BLS(f, grad, @(alpha)xk + alpha.*pk, 1e-4, 0.5, 1);
    end
    x_next = xk + alpha.*pk;
    grad_next = grad(x_next)';
    yk = grad_next - grad_k;

    % if k exceeds the available memory, update the arrays inside it
    [s,y] = memory_handling(s, y, xk, x_next, yk, k, l);

    % update the parameters
    grad_k=grad_next;
    xk=x_next;
    k=k+1;

    % if the last gradient didn't have a great change from the previous one
    norm_y = norm(y(end));
end
end

function [s, y] = memory_handling(s, y, xk, x_next, yk, k, l)
%{
Handles the memory available to the LBFGS method, where l is the size of
the former.
Input:
    s: (matrix) l displacements defined as y_k=x_{k+1}-x_k, each displacement
    is a column
    y: (matrix) l differences between gradients defined as y_k=\nabla
    xk: (array) current point
    x_next: (array) new point
    yk: (array) difference between gradient(x_next) and gradient(xk)
    k: (int) current iteration
    l: (int) number of steps to keep in memory
Output:
    s: (matrix) updated matrix of the displacements, where the most recent
    one is in the last column
    y: (matrix) updated matrix of the differences between gradients, where
    the most recent one is in the last column
%}

    if k > l
        s = [s(:, 2:end) x_next-xk];
        y = [y(:, 2:end) yk];
    else
        s= [s x_next-xk];
        y = [y yk];
    end
end