function a = ArmijoWolfe(f,grad,pk,xk)
max_f_eval=1000;
m1=1e-4;
m2=0.9;
a_start=1;
sfgrd=0.01;
tau=0.9;
min_a=1e-16;
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
    as = as/tau;
    ls_iter=ls_iter+1;
end
am = 0;
a = as;
phi_pm = phi_p0;
% ls_iter=1;
while ls_iter <= max_f_eval && as - am > min_a && phi_ps > 1e-12
    % compute the new value by safeguarded quadratic interpolation
    a = (am * phi_ps - as * phi_pm) / (phi_ps - phi_pm);
    %a = max(am * (1 + self.sfgrd), min(_as * (1 - self.sfgrd), a))
    a = max(am + (as - am) * sfgrd, min(as - (as - am) * sfgrd, a));

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