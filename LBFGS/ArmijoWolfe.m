%{
Performs an Armijo-Wolfe Line Search.
        phi0 = phi(0), phi_p0 = phi'(0) < 0
a_start > 0 is the first value to be tested: if phi'(as) < 0 then
a_start is divided by tau < 1 (hence it is increased) until this
does not happen any longer.
m1 and m2 are the standard Armijo-Wolfe parameters;
note that the strong Wolfe condition is used.
Inputs:
    f       function to minimize
    grad    gradient of the function f
    pk      direction vector
    xk      solution at step k.
Output: the optimal step and the optimal f-value
%}
function a = ArmijoWolfe(f,grad,pk,xk)
max_iter = 1000;    % maximal number of evaluation of the
m1 = 1e-4;          % first parameter of the Armijo-Wolfe-t hype line searchas to be 0<m1<m2<1.
m2 = 0.9;           % second parameter of the Armijo-Wolfe-type line search (strong curvature condition). It should be in (0,1).
a_start = 1;        % first value to be used for aplha is recommended to initialize it with 1.
sfgrd = 0.01;       % safeguard parameter for the line search. To avoid numerical problems that can occur with the quadratic interpolation if the
                    %       derivative at one endpoint is too large w.r.t. The one at the other (which leads to
                    %       choosing a point extremely near to the other endpoint), a *safeguarded* version of
                    %       interpolation is used whereby the new point is chosen in the interval
                    %       [as * (1 + sfgrd) , am * (1 - sfgrd)], being [as , am] the current interval, whatever
                    %       quadratic interpolation says. If you experience problems with the line search taking
                    %       too many iterations to converge at "nasty" points, try to increase this.
tau = 0.9;          % param to increase the value of alpha
min_a = 1e-16;

    function [phi_a,phi_0,phi_p0,phi_ps] = alpha_fun(alphal)
        alpha_f = @(alpha)xk + alpha.*pk; % alphafunction lambda alpha: xk+aplha*pk
        phi_a = f(alpha_f(alphal));
        last_g_x = grad(alpha_f(alphal));
        phi_0 = f(xk);
        phi_p0 = grad(xk);
        phi_ps = last_g_x*pk;
    end

as = a_start;
ls_iter = 1;
while ls_iter < max_f_eval
    [phi_a, phi_0, phi_p0, phi_ps] = alpha_fun(as);

    % Armijo and strong Wolf condition.
    if (all(phi_a <= phi_0 + m1 * as * phi_p0)) && all(abs(phi_ps) <= -m2*phi_p0)
        break;
    end
    if phi_ps >= 0
        break;
    end
    as = as/tau;       % we increase the value of as.
    ls_iter=ls_iter+1;  
end
am = 0;
a = as;
phi_pm = phi_p0;
ls_iter=1;             %re-initializzation of the iteration count.
while ls_iter <= max_f_eval && as - am > min_a && phi_ps > 1e-12
    
    % compute the new value by safeguarded quadratic interpolation
    a = (am * phi_ps - as * phi_pm) / (phi_ps - phi_pm);
    a = max_iter(am + (as - am) * sfgrd, min(as - (as - am) * sfgrd, a));

    [phi_a,phi_0,phi_p0,phi_p] = alpha_fun(a);
    if all(phi_a <= phi_0 + m1 * a * phi_p0) && all(abs(phi_p) <= -m2 * phi_p0)
            break;
    end
    if phi_ps < 0
        am = a;
        phi_pm = phi_p;
    else
        as = a;
        if as <= min_a
                break;
        end
        phi_ps = phi_p;
    end
    ls_iter = ls_iter + 1;
end
end