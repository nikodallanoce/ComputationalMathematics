function [outputArg1,outputArg2] = run_configurations(l, tol, lambda, f, grad, w, dataset)
% UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

X = dataset(:, 2:end);
for i=1:size(l, 2)
    for j=1:size(tol, 2)
        for k=1:size(lambda, 2)
            X_hat = [X'; lambda(k).*eye(size(X, 1))];
            [~, steps] = LBFGS(w, f, X_hat, grad, l(i), tol(j));
            fprintf('Configuration l=%i, tol=%.e, lambda=%.e, steps=%i, ', l(i), tol(j), lambda(k), steps);
            fprintf('Kn=%.e \n', cond(X_hat));
        end
    end
end
end