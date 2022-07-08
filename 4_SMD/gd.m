function [x, k] = gd(func, grad, x0, b, tol, eta, alpha)

k=0;
x = x0;
f=func(x);
df = grad(x)';
old_delta_x = 0;

while(norm(df)>tol && k<100)

    
end

end