clear;
%m=1200; n=200;
%A = rand(m, n);
%B=A;

dataset = readtable("../datasets/ML-CUP21-TR.csv");
dataset = table2array(dataset);
X = dataset(:, 2:end);

% Build \hat_{X} and \hat_{y}
[m, n0] = size(X);
lambda = 1e-4;
X = [X'; lambda.*eye(m)];
[m, n] = size(X);
y = [randn(n0, 1); zeros(m-n0, 1)];
A=X;
%b= rand(m,1);
%[Q, R] = qr(A);
[Q1, R1] = thinqr(A);
x = linsolve(R1, Q1' *y);
[Q_t, R_t] = qr(A, "econ");
x_t= A\y;
disp(norm(x-x_t));
disp(norm(A*x - y));
disp(norm(A*x_t - y));
%syms a b c d e f g h i j k l m;
%Q = [[1 0 0 0];[0 1 0 0];[0 0 a b];[0 0 c d]];
%H = [[1 0 0 0];[0 e f g];[0 h i j];[ 0 k l m]];
%[Qm, Rm] = myqr(A);
%fprintf("Q error: %e, R error: %e", sum(abs(Q-Qm), 'all'), sum(abs(R-Rm),'all'));