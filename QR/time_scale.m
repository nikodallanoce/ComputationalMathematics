clear;
format long e;
rng(1);
lambda = 1e-4;
runs = 20;
sol = zeros(1,runs);
x_axis = zeros(1,runs);
for i=1:runs
    m=1000+(i-1)*250;
    x_axis(i) = m;
    n = 50;
    A = rand(m, n);
    y = randn(m, 1);
    for j=1:10
        [Q1y, R1] = thinqr(A, y);
        [Q_star, R_star] = qr(A, "econ");
        x = linsolve(R1, Q1y);
        x_star = linsolve(R_star, Q_star'*y);
        sol(i) = sol(i)+norm(x - x_star);
    end
    sol(i) = sol(i)/10;
    disp(i);
end
plot(x_axis, sol);
xlabel('matrix rows');
ylabel('average time (seconds)');
title('Thin QR average completion time by varying number of rows');
grid on;