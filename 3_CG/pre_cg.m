function [x_k, k] = pre_cg(A, M, x_0, b, tol)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

x_k = x_0;
r_k = A *x_0 -b;
y_k = M\r_k;
p_k = -y_k;

k=0;

while(norm(r_k)>tol && k<1000)
    
    [x_k, r_k, p_k, y_k] = iteration(A, M, r_k, p_k, x_k, y_k);
        k=k+1;
        if(mod(k,5)==0)
            disp(k);
        end
    k = k+1;
end


end

function [x_k_next, r_k_next, p_k_next, y_k_next] = iteration(A, M, r_k, p_k, x_k, y_k)

    a_k = (r_k'*y_k)/(p_k'*A*p_k);
    x_k_next = x_k + a_k*p_k;
    r_k_next = r_k +a_k * A * p_k;
    y_k_next = M\r_k_next;
    B_k_next = (r_k_next' * y_k_next) / (r_k'*y_k);
    p_k_next = -y_k_next + B_k_next*p_k;
end