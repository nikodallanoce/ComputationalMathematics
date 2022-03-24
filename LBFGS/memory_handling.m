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

