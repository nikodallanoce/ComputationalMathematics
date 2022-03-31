function [Q, R] = myqr(A)
[m, n] = size(A);
Q = eye(m);
H_h = {};
for j = 1:n
    u = householder_vector(A(j:end, j));
    H = eye(length(u)) - 2*u*u';
    H_h = [H_h H];
    A(j:end,j:end) = H * A(j:end,j:end);
    %Q(:, j:end) = Q(:, j:end) * H;
end
R = A;

H = cell2mat(H_h(end));
[mh, nh] = size(H);
Q=eye(m,n);
Q(end-mh+1:end, end-nh+1:end)= H;
[mq, nq] = size(Q);
Q = Q*eye(nq, n);
[~,l] = size(H_h);
for i = l-1: -1: 1
    I=eye(m);
    H=cell2mat(H_h(i));
    [mh, nh] = size(H);
    I(end-mh+1:end, end-nh+1:end)= H;
    Q = I*Q;
end
end

