function [s,y] = memory_handling(s, y, xk, x_next, yk, k, l)

    if k > l
        s = [s(:,2:end) x_next-xk];
        y = [y(:, 2:end) yk];
    else
        s= [s x_next-xk];
        y = [y yk];
    end
end

