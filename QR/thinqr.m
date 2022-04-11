function [Q1, R1] = thinqr(A)
[m, n] = size(A);
U = {};
for j = 1:min(m-1,n)
    [u, s] = householder_vector(A(j:end, j));
    U = [U u];
    A(j,j) = s;
    A(j+1:end,j) = 0;
    A(j:end,j+1:end) = A(j:end,j+1:end) - 2*u*(u'*A(j:end,j+1:end));
end
R1=A(1:n,:);
Q1 = compute_Q(U, m, n);
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

