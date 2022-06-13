clear;
format long e;
%rng(1);
dataset = readtable("../datasets/ML-CUP21-TR.csv");
dataset = table2array(dataset);
X = dataset(:, 2:end);

% Build \hat_{X} and \hat_{y}
[m0, n0] = size(X);
lambdas = [1, 1e-2, 1e-4, 1e-8];


runs = 10;
res = zeros(length(lambdas), runs);
err = zeros(length(lambdas), runs);

for i=1:length(lambdas)
    lambda = lambdas(i);
    A = [X'; lambda.*eye(m0)];
    [m, n] = size(A);
    for j=1:runs
        y = [randn(n0, 1); zeros(m-n0, 1)];
        tic;
        [Q1y, R1] = thinqr(A, y);
        x = linsolve(R1, Q1y);
        elapsed = toc;
        [Q_t, R_t] = qr(A, "econ");
        x_t= linsolve(R_t, Q_t'*y);
        
        res(i,j) = norm(x - x_t);
        %disp(norm(x - x_t));
        %disp(norm(A*x - y)/norm(y));
        err(i,j) = norm(A*x - y)/norm(y);
        disp(elapsed);
    end
end
