function [Q1y, R1] = thinqr(A, y)
[m, n] = size(A);
Q1y=y;
for j = 1:min(m-1,n)
    [u, s] = householder_vector(A(j:end, j));
    A(j,j) = s;
    A(j+1:end,j) = 0;
    A(j:end,j+1:end) = A(j:end,j+1:end) - 2*u*(u'*A(j:end,j+1:end));
    Q1y(j: end) = Q1y(j:end) - 2*u*(u' * Q1y(j:end));
end
R1=A(1:n,:);
Q1y = Q1y(1:n);
end