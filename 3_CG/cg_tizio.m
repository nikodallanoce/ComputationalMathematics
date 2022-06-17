function [xk, i] = cg_tizio(x0, A, b, tol)
    %{
    Conjugate gradient algorithm

    Parameters:
    x0: initial point
    A: matrix of interest
    b: vector in linear system (Ax = b)
    max_iter: max number of iterations to run CG
    %}

    % initial iteration
    xk = x0;
    rk = A * xk - b;
    pk = -rk;

    max_iter = 500;

    for i=1:max_iter

        % run conjugate gradient iteration
        [xk, rk, pk] = iterate_CG(xk, rk, pk, A);

        % compute absolute error and break if converged
        %err = sum(abs.(rk)); push!(errors, err)
        if (norm(rk) < tol)
            break;
        end
    end
    return
end

function [xk_new, rk_new, pk_new] = iterate_CG(xk, rk, pk, A)
    %{
    Basic iteration of the conjugate gradient algorithm

    Parameters:
    xk: current iterate
    rk: current residual
    pk: current direction
    A: matrix of interest
    %}

    %construct step size
    ak = (rk' * rk) / (pk' * A * pk);

    % take a step in current conjugate direction
    xk_new = xk + ak * pk;

    % construct new residual
    rk_new = rk + ak * A * pk;

    % construct new linear combination
    betak_new = (rk_new' * rk_new) / (rk' * rk);

    % generate new conjugate vector
    pk_new = -rk_new + betak_new * pk;

    return
end