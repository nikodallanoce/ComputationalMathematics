clear;
A = rand(10, 8);
[Q, R] = qr(A);
[Q1, R1] = myqr(A);
fprintf("Q error: %e, R error: %e", sum(abs(Q-Qm), 'all'), sum(abs(R-Rm),'all'));