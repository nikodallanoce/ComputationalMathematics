clear;

addpath ../utilities;

%lambdas = [1e5, 1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5];
lambdas = [1e4, 1e2, 1e0, 1e-2, 1e-4];
times = zeros(length(lambdas),1);
errors = [];
ks = zeros(length(lambdas),1);
for i=1:length(lambdas)
    [X_hat, y_hat, w, w_star] = build_matrices("../datasets/ML-CUP21-TR.csv", lambdas(i));
    %rmpath ../utilities;
  
    % Compute the solution using conjugate gradient

    b = X_hat' * y_hat;

    %fprintf("%e %e\n", lambdas(i), cond(A))
    %disp([lambdas(i), cond(A)])

    x0 = zeros(length(w_star),1);
    
    tol = 1e-14;
    
    time_elapsed = tic;
    [x, k, err] = cg_opt(sparse(X_hat), x0, b, tol, w_star);
    time_elapsed = toc(time_elapsed);
    times(i) = time_elapsed;
    errors = [errors err];
    ks(i) = k;
end

linear = zeros(length(max(ks)), 1);
linear(1) = max(errors);
for i = 2:max(ks)
    linear(i) = linear(1)/pow2(i-1);
end
s = 1;
e = ks(1);
semilogy(errors(s:e),'LineWidth',2);
hold on;
for i=2:length(ks)
    s = e+1;
    e = ks(i) + s - 1;
    semilogy(errors(s:e), 'LineWidth',2);
end
semilogy(linear, 'LineWidth',2);
hold off;
grid on;

x_lab = strings(length(lambdas)+1,1);
for i=1:length(lambdas)
    x_lab(i)=num2str(lambdas(i),"%.1e");
end
x_lab(end) = "linear";
legend(x_lab);
title("||w-w*|| by varying lambda values")
xlabel("iterations");
ylabel("Log10 error")

%plot(times);
