clear;
rng(1);
addpath ../1_LBFGS/;
addpath ../2_QR/;
addpath ../3_CG/;
addpath ../4_SMD/;
runs = 20;
x_axis = zeros(1, runs);
lbfgs_times = zeros(1, runs);
qr_times = zeros(1, runs);
cg_times = zeros(1, runs);
smd_times = zeros(1, runs);
for i=1:runs
    m = 1000+(i-1)*250;
    x_axis(i) = m;
    n = 200+(i-1)*50;
    A = rand(m, n);
    y = randn(m, 1);
    x0 = zeros(n, 1);
    x_star = A\y;
    resid_fun = @(xk) norm(A*xk-y)/norm(y);
    for j=1:5
        % compute LBFGS execution time
        start = tic;
        [~, ~, ~, ~] = LBFGS(x0, A, A'*y, y, 20, 1e-14, false, x_star);
        lbfgs_times(1, i) = lbfgs_times(1, runs) + toc(start);

        % compute thin QR execution time
        start = tic;
        [~, ~] = thinqr(A, y);
        qr_times(1, i) = qr_times(1, i) + toc(start);

        % compute conjugate gradient execution time
        start = tic;
        [~, ~, ~] = cg_opt(sparse(A), x0, A'*y, 1e-14, x_star);
        cg_times(1, i) = cg_times(1, i) + toc(start);

        % compute standard momentum descent execution time
        start = tic;
        [~, ~, ~, ~] = mgd_eqn(A, x0, x_star, resid_fun, 1e-12, 0, A'*y, 1e4, false, false);
        smd_times(1, i) = smd_times(1, i) + toc(start);
    end
    fprintf("Matrix %i completed\n", i);
    lbfgs_times(1, i) = lbfgs_times(1, i)/5;
    qr_times(1, i) = qr_times(1, i)/5;
    cg_times(1, i) = cg_times(1, i)/5;
    smd_times(1, i) = smd_times(1, i)/5;
end
rmpath ../1_LBFGS/;
rmpath ../2_QR/;
rmpath ../3_CG/;
rmpath ../4_SMD/;
plot(x_axis, lbfgs_times);
hold on;
plot(x_axis, qr_times);
plot(x_axis, cg_times);
plot(x_axis, smd_times);
legend(["L-BFGS", "thin QR", "CG", "SMD"]);
xlabel('matrix rows');
ylabel('average time (seconds)');
title('Execution time by varying both sizes');
hold off;
grid on;