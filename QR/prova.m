clear;
format long e;
%m=1200; n=200;
%A = rand(m, n);
%B=A;
rng(1);
dataset = readtable("../datasets/ML-CUP21-TR.csv");
dataset = table2array(dataset);
X = dataset(:, 2:end);

% Build \hat_{X} and \hat_{y}
[m, n0] = size(X);
lambda = 1e-8;
X = [X'; lambda.*eye(m)];
[m, n] = size(X);
y = [randn(n0, 1); zeros(m-n0, 1)];
A=X;
%b= rand(m,1);
%[Q, R] = qr(A);
[Q1y, R1] = thinqr(A, y);
x = linsolve(R1, Q1y);
[Q_t, R_t] = qr(A, "econ");
x_t= A\y;
disp(norm(A*x - y));
disp(norm(A*x - y)/norm(y));
disp(norm(A*x_t - y)/norm(y));