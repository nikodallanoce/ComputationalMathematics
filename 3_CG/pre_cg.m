function [x_k, k] = pre_cg(A, x_0, b, tol, M)
    
    r_k = A*x_0 - b;
    y_k = M\r_k;
    p_k = -y_k;
    x_k = x_0;
    k = 0;
    tolb = tol*norm(b);

    while(norm(r_k)>tolb)
              
        [x_k, r_k, p_k, y_k] = iteration(A, r_k, p_k, x_k, y_k, M);
        k=k+1;

        if(mod(k,5)==0)
            %norm(r_k)
        end
       
    end
end

function [x_k_next, r_k_next, p_k_next, y_k_next] = iteration(A, r_k, p_k, x_k, y_k, M)
    
    %r_k_q = r_k'*y_k;
    %A_pk = A*p_k;
    a_k = r_k' * y_k /(p_k' * A * p_k);   
    x_k_next = x_k + a_k * p_k;
    r_k_next = r_k + a_k * A * p_k;
    y_k_next = M\r_k_next;
    B_k_next = r_k_next' * (y_k_next - y_k) / (r_k' * y_k);
    p_k_next = -y_k_next + B_k_next * p_k;
end