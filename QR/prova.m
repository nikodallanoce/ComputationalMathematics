clear;
A = rand(10, 8);
[Q, R] = qr(A);
[Qm, Rm] = myqr(A);
MM=Qm*Rm;
fprintf("Q error: %e, R error: %e", sum(abs(Q-Qm), 'all'), sum(abs(R-Rm),'all'));