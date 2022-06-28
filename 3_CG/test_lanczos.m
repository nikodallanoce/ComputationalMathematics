% Written by Danny Bickson, CMU
% Matlab code for running Lanczos algorithm
clear;
clc;

m = 12;
n = 8;
A = rand(m,n);
b = rand(m,1);
A_n=A'*A;

A_aux = [eye(m, m), A; A', zeros(n, n)];
[m_aux, n_aux] = size(A_aux);
b_aux = [b; zeros(m_aux-length(b),1)];

%disp(['eigenvalues are ', num2str(flipud(eig(A_n))')]);
%[T, V] = lanczos(A_n,n);
%A_tilde = V*T*V';

ris_aux = A_aux\b_aux;
x_aux = ris_aux(end-n+1:end);

[Ta, Va] = lanczos(A_aux, n-1);

disp(norm(A\b - x_aux))

[L, U ] = lu(A_n);