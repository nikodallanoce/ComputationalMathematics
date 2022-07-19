function [Q1y, R1] = thinqr(A, y)
% Thin QR factorization
% Inputs:
%       A           input matrix
%       y           array of expected values
%
% Output:
%       Q1y         product between the orthogonal matrix Q1 and y,
%                   if y is not passed then only Q1 is returned
%       R1          upper triangular matrix
%
% Reference:
%       Algorithm 4 from our report.
%
% Created by Niko Dalla Noce, Alessandro Ristori and Simone Rizzo

y_exists = false;
[m, n] = size(A);

if exist("y", "var")
    y_exists = true;
    Q1y=y;
else
    U = {};
    fprintf("No expected values were passed, Q1 will be fully computed\n");
end
for j = 1:min(m-1, n)
    [u, s] = householder_vector(A(j:end, j));
    A(j, j) = s;
    A(j+1:end, j) = 0;
    A(j:end, j+1:end) = A(j:end, j+1:end) - 2*u*(u'*A(j:end, j+1:end));
    if y_exists
        Q1y(j: end) = Q1y(j:end) - 2*u*(u'*Q1y(j:end));
    else
        U = [U u];
    end
end
R1 = A(1:n,:);
if y_exists
    Q1y = Q1y(1:n);
else
    Q1y = compute_Q1(U, m, n);
end
end

function Q1 = compute_Q1(U, m, n)
% Compute orthogonal matrix Q1
% Inputs:
%       U           array of householder vectors
%       m           number of rows
%       n           number of columns
%
% Output:
%       Q1         orthogonal matrix
%
% Reference:
%       Algorithm 3 from our report.
%
% Created by Niko Dalla Noce, Alessandro Ristori and Simone Rizzo

u = cell2mat(U(end));
mu = length(u);
Q1 = eye(m, n);

H = 2*u*u(1);

Q1(end-mu+1:end, end)= Q1(end-mu+1:end, end)- H;
[~,l] = size(U);

for i = l-1: -1: 1
    u = cell2mat(U(i));
    Q1(i:end, i:end) =  Q1(i:end, i:end) - 2*u*(u'*Q1(i:end, i:end));
end
end