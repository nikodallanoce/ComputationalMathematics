function r = compute_direction(gradient, s, y, H0, k)
%{
Computes the search direction, p, for the current iteration of the LBFGS
method.
Input:
    gradient: (array) gradient of the function evaluated at the current point
    s: (matrix) l displacements defined as y_k=x_{k+1}-x_k, each displacement
    is a column
    y: (matrix) l difference between gradients defined as y_k=\nabla
    f_{k+1}-\nabla f_{k}, each element is a column
    I: (matrix) nxn identity matrix
    k: (int) current iteration
Output:
    r: (array) current search direction defined as H_k \nabla f_k
%}

q = gradient;
[~, nc] = size(s);
if k <= nc
    nc = k-1;
end
alpha = zeros(nc);
rho = zeros(nc);
for i = nc:-1:1
    rho(i) = 1/(y(:, i)'*s(:, i));
    alpha(i) = rho(i).* s(:, i)' * q;
    q = q - alpha(i).* y(:, i);
end

gamma = 1;
if k > 1
    gamma = s(:, nc)'*y(:, nc) / norm(y(:, nc))^2;
end

H0 = gamma * H0;
r = H0 * q;

for i = 1:nc
    beta= rho(i) * y(:, i)' * r;
    r = r + s(:, i)*(alpha(i) - beta);
end
end
