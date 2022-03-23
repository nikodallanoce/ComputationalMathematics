function r = compute_direction(gradient, s, y, n, k)

q = gradient;
[~,nc] = size(s);
alpha = zeros(nc);
rho = zeros(nc);

for i = nc: -1: 1
    rho(i) = 1/(y(:,i)'*s(:,i));
    alpha(i) = rho(i).* s(:,i)' * q;
    q = q - alpha(i).* y(:,i);
end

gamma = 1;
if(k>0)
    gamma = s(:,nc)'*y(:,nc) / norm(y(:,nc))^2;
end

H0 = gamma * eye(n);
r = H0 * q;

for i = 1:nc
    %rho = 1/(y(:,i)'*s(:,i));
    beta= rho(i) * y(:,i)' * r;
    %alpha = rho * s(:,i)' * q;
    r = r + s(:,i)*(alpha(i) - beta);
end
end
