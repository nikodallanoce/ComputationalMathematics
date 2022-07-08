function [x_k, k, errors] = cg_opt(A, x_0, b, tol, x_star)

    r_k = A'*(A*x_0) - b;
    p_k = -r_k;
    x_k = x_0;
    k = 0;
    tolb = tol;%*norm(b);
    errors = [];

    while(norm(r_k)>tolb)
        
        [x_k, r_k, p_k] = iteration(A, r_k, p_k, x_k);
        errors = [errors norm(x_star - x_k)];
        k=k+1;

        if(mod(k,5)==0)
            %disp(k);
        end
    end
end

function [x_k_next, r_k_next, p_k_next] = iteration(A, r_k, p_k, x_k)
    r_k_q = r_k'*r_k;
    A_pk = A*p_k;
    a_k = r_k_q/(A_pk'*A_pk);
    x_k_next = x_k + a_k * p_k;
    r_k_next = r_k + A' * a_k * A_pk;
    B_k_next = (r_k_next' * r_k_next) / r_k_q;
    p_k_next = -r_k_next + B_k_next*p_k;
end