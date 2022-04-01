function [Q1, R1] = thinqr_o(A)
[m, n] = size(A);
U = {};
for j = 1:min(m-1,n)
    [u, s] = householder_vector(A(j:end, j));
    U = [U 2*(u*u')];
    A(j,j) = s;
    A(j+1:end,j) = 0;
    A(j:end,j+1:end) = A(j:end,j+1:end) - 2*u*(u'*A(j:end,j+1:end));
    %Q(:, j:end) = Q(:, j:end) - Q(:,j:end)*u*2*u';
end
R1=A(1:n,:);
Q1 = compute_Q(U, m, n);
end

function Q = compute_Q(H_h, m, n)
H = cell2mat(H_h(end));
[mh, nh] = size(H);
Q=eye(m);

Q(end-mh+1:end, end-nh+1:end)= Q(end-mh+1:end, end-nh+1:end)-H;
Q = Q*eye(m, n);
[~,l] = size(H_h);
H_i=eye(m);

for i = l-1: -1: 1
    H = cell2mat(H_h(i));
    [mh, nh] = size(H);
    %H_i(end-mh+1:end, end-nh+1:end)= H_i(end-mh+1:end, end-nh+1:end)-H;
    %Q1 = Q(i:end, i:end);
    %Q(i:end, i:end) =  Q(i:end, i:end) - 2* u(u'*Q(i:end, i:end));
    Q(i:end, i:end) = (eye(mh)-H) * Q(i:end, i:end);
    %Q = H_i*Q;
    %Q(1:end-i, end-nh+1:end) = (eye(mh)-H) * Q(end-mh+1:end, end-nh+1:end);
    %H_i(end-mh+1:end, end-nh+1:end) = eye(mh, nh);
    
end
end

