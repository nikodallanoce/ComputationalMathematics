clear;
rng(1);

micheli = readtable('ML-CUP21-TR.csv');
micheli = table2array(micheli);
micheli = micheli(:, 2:end);

[m, n0] = size(micheli);
micheli_hat = [micheli'; eye(m)];

[m, n] = size(micheli_hat);

%m = 10000;
%n = 100;
%X = randn(m, n);
X = micheli_hat;
w = randn(n, 1);
z = [randn(n0,1); zeros(m-n0,1)];
%grad = 2.*w'*(X'*X) - 2.*y'*X;
grad = @(x) 2.*x'*X'*X - 2.*z'*X;
f= @(x) x'*X'*X*x - 2.*z'*X*x+z'*z;

l = 15;

ris = X\z;

[w, k] = LBFGS(w, f, X, grad, l, 1e-3);
%disp(k);
%r = compute_direction(grad', s, y, n, 3);