function run_configurations(l, lambda, f, grad, y_hat, w, dataset)
% UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

X = dataset(:, 2:end);
for i=1:size(l, 2)
    for j=1:size(lambda, 2)
        X_hat = [X'; lambda(j).*eye(size(X, 1))];
        [output_bls, steps_bls] = LBFGS(w, f, X_hat, grad, l(i), 1e-8, false);
        [output_wolfe, steps_wolfe] = LBFGS(w, f, X_hat, grad, l(i), 1e-8, true);
        fprintf('Configuration l=%i, lambda=%.e, Kn=%.e\n', l(i), lambda(j), cond(X_hat));
        fprintf('Error Armijo %.e, steps Armijo %i\n', norm(X_hat*output_bls-y_hat), steps_bls);
        fprintf('Error Armijo-Wolfe %.e, steps Armijo-Wolfe %i\n\n', norm(X_hat*output_wolfe-y_hat), steps_wolfe);
    end
end
end