function [x_k, k] = cg(A, x_0, b, tol)

    r_k = A*x_0 - b;
    p_k = -r_k;
    x_k = x_0;
    k = 0;

    while(norm(r_k)>tol)
        
        [x_k, r_k, p_k] = iteration(A, r_k, p_k, x_k);
        k=k+1;

        if(mod(k,5)==0)
            disp(k);
        end
    end
end

function [x_k_next, r_k_next, p_k_next] = iteration(A, r_k, p_k, x_k)

    a_k = (r_k'*r_k)/(p_k'*A*p_k);
    x_k_next = x_k + a_k*p_k;
    r_k_next = r_k +a_k * A * p_k;
    B_k_next = (r_k_next' * r_k_next) / (r_k'*r_k);
    p_k_next = -r_k_next + B_k_next*p_k;
end