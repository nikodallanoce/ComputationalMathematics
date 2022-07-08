function [Q1y, R1] = thinqr(A, y)
y_exists = false;
[m, n] = size(A);

if exist("y", "var")
    y_exists = true;
    Q1y=y;
else
    U = [];
end
for j = 1:min(m-1,n)
    [u, s] = householder_vector(A(j:end, j));
    A(j,j) = s;
    A(j+1:end,j) = 0;
    A(j:end,j+1:end) = A(j:end,j+1:end) - 2*u*(u'*A(j:end,j+1:end));
    if y_exists
        Q1y(j: end) = Q1y(j:end) - 2*u*(u' * Q1y(j:end));
    else
        U = [U u];
    end
end
R1=A(1:n,:);
if y_exists
    Q1y = Q1y(1:n);
else
    Q1y = compute_Q(U, m, n);
end
end

function Q = compute_Q(U, m, n)
u = cell2mat(U(end));
mu = length(u);
Q = eye(m,n);

H=2*u*u(1);

Q(end-mu+1:end, end)= Q(end-mu+1:end, end)- H;
[~,l] = size(U);

for i = l-1: -1: 1
    u = cell2mat(U(i));
    Q(i:end, i:end) =  Q(i:end, i:end) - 2* u*(u'*Q(i:end, i:end));
end
end