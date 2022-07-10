clear;

lambdas = [1e4, 1e2, 1, 1e-2, 1e-4];
addpath ../utilities;
for i=1:length(lambdas)
    [X_hat, y_hat] = build_matrices("../datasets/ML-CUP21-TR.csv", lambdas(i));

    % Compute our thin QR and its accuracy
    [Q1, R1] = thinqr(X_hat);
    accuracy = norm(X_hat-Q1*R1)/norm(X_hat);
    fprintf("Lambda: %e\nOur thin qr accuray: %e\n", lambdas(i), accuracy);

    % Compute matlab thin QR and its accuracy
    [Q1_m, R1_m] = qr(X_hat, "econ");
    accuracy_m = norm(X_hat-Q1_m*R1_m)/norm(X_hat);
    fprintf("Matlab thin qr accuray: %e\nDistance between accuracies: %e\n\n", accuracy_m, abs(accuracy-accuracy_m));
end
rmpath ../utilities;