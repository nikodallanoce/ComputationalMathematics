clear;
addpath ../utilities;

lambdas = zeros(52,1);
tol = 1e-16;
times = zeros(length(lambdas),1);
conds = zeros(length(lambdas),1);
for i=1:length(lambdas)
    lambdas(i) = power(10,(2-(i/4)));
    [X_hat, y_hat, w, w_star] = build_matrices("../datasets/ML-CUP21-TR.csv", lambdas(i));
    fprintf("%2d, %5.4e, %5.5e\n", i, lambdas(i), cond(X_hat));
    b = X_hat' * y_hat;
    x0 = zeros(length(b),1);
    t_start = tic;
    [x, k, err] = cg_opt(sparse(X_hat), x0, b, tol, w_star);
    times(i) = toc(t_start);
    conds(i) = cond(X_hat);
end
hold on;
plot(times);
hold off;
leg = strings(length(lambdas),1);
for i=1:times
    leg(i) = num2str(cond(i), "%.3e");
end

%legend(leg);
plot(times, '-o')
title("CG execution time by varying $\kappa(\hat{X})$", 'Interpreter','latex');
x_tick = [sqrt(conds(1)), sqrt(conds(10)), sqrt(conds(20)), sqrt(conds(30)), sqrt(conds(40)), sqrt(conds(50))];
set(gca, 'XTickLabel', x_tick)
xlabel("$\kappa(\hat{X})$", 'Interpreter','latex');
ylabel("time $\mathcal{O}(m*\sqrt{\kappa(\hat{X})})$", 'Interpreter','latex');


