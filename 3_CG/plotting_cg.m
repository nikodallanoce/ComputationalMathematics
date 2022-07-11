clear;
addpath ../utilities;
%lambdas = [1e5, 1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5];
lambdas = [1e4, 1e2, 1e0, 1e-2, 1e-4];
times = zeros(length(lambdas),1);
errors = {};
ks = zeros(length(lambdas),1);
for i=1:length(lambdas)
    [X_hat, y_hat, w, w_star] = build_matrices("../datasets/ML-CUP21-TR.csv", lambdas(i));
  
    % compute the solution using conjugate gradient
    b = X_hat' * y_hat;

    %fprintf("%e %e\n", lambdas(i), cond(X_hat))
    %disp([lambdas(i), cond(A)])

    x0 = zeros(length(w_star), 1); 
    
    time_elapsed = tic;
    [x, k, err] = cg_opt(sparse(X_hat), x0, b, 1e-14, w_star);
    time_elapsed = toc(time_elapsed);
    times(i) = time_elapsed;
    errors = [errors err/norm(w_star)];
    ks(i) = k;
end
rmpath ../utilities;
% build rates
linear = zeros(length(max(ks)), 1);
max_err = 0;
for i=1:length(errors)
    curr_errors = cell2mat(errors(i));
    if max(curr_errors)>max_err
        max_err = max(curr_errors);
    end
end

[m,n] = size(errors);
err1 = cell2mat(errors(1,1));
err1 = err1(1);
quad = err1;
lin = zeros(size(errors));
lin(1) = err1;
for i = 2:20
    lin(i) = lin(1)/(2^(i-1));
    if (i<8)
        quad = [quad quad(1)/power(2, 2^(i-1))];
    end
end

% plot errors
for i=1:length(ks)
    if (i == 1)
        semilogy(lin, 'LineWidth', 1);
        hold on;   
        semilogy(quad, 'LineWidth', 1);
    end
    semilogy(cell2mat(errors(i)), 'LineWidth', 1);
end
hold off;
grid on;

% insert labels
labels = strings(length(lambdas), 1);
%labels(1) = "linear";
for i=1:length(lambdas)
    labels(i) = num2str(lambdas(i), "%.1e");
end

legend(["linear"; "quadratic"; labels], "Location", "southeast");
title("CG convergence speed by varying lambda values")
xlabel("steps");
ylabel("$\frac{||w - w^{*}||}{ ||w^{*}||}$", 'Interpreter','latex');
