clear;
format long e;
%rng(1);
dataset = readtable("../datasets/ML-CUP21-TR.csv");
dataset = table2array(dataset);
X = dataset(:, 2:end);

% Build \hat_{X} and \hat_{y}
[m0, n0] = size(X);
lambdas = [1e4, 1e2, 1, 1e-2, 1e-4];

runs = 3;
res = zeros(length(lambdas), runs);
err = zeros(length(lambdas), runs);
grad = zeros(length(lambdas), runs);

for i=1:length(lambdas)
    lambda = lambdas(i);
    X_hat = [X'; lambda.*eye(m0)];
    [m, n] = size(X_hat);
    XtX = X_hat' * X_hat;
    
    for j=1:runs
        y = [randn(n0, 1); zeros(m-n0, 1)];
        ytX2 = 2.*y'* X_hat;
        yty = y'*y;
        grad_lls = @(w)2.*w'*XtX-ytX2;
        tic;
        [Q1y, R1] = thinqr(X_hat, y);
        x = linsolve(R1, Q1y);
        elapsed = toc;
        %[Q_t, R_t] = qr(A, "econ");
        %x_t= linsolve(R_t, Q_t'*y);
        x_t = X_hat\y;
        err(i,j) = norm(x - x_t)/norm(x_t);
        %disp(norm(x - x_t));
        %disp(norm(A*x - y)/norm(y));
        res(i,j) = norm(X_hat*x - y)/norm(y);
        grad(i,j) = norm(grad_lls(x));
        disp(elapsed);
    end
end
