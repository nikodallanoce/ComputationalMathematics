function [residues, errors, times, iters, config] = run_configurations_lbfgs(runs, l, lambda, tol, D, y, Wolfe)
tot_runs = length(lambda)*length(l);
residues = -ones(tot_runs, runs, 1000);
errors = -ones(tot_runs, runs, 1000);
times = zeros(tot_runs, runs);
iters = zeros(tot_runs, runs);
config = strings(tot_runs,1);
[m0, ~] = size(D);
% Compute LBFGS
X = [D'; eye(m0)];
[~, n] = size(X);
for r=1: runs
    fprintf("Run %d \n \n", r);
    w = randn(n, 1);
    ind = 1;
    for i=1:length(l)
        curr_l=l(i);
        for j=1:length(lambda)
            curr_lambda = lambda(j);
            X = [D'; curr_lambda.*eye(m0)];
            [~, n] = size(X);
            XtX = X'*X;
            ytX2 = 2.*y'*X;
            yty = y'*y;
            grad_lls = @(x) 2.*x'*XtX - ytX2;
            f_lls = @(x) x'*XtX*x - ytX2*x + yty;
            x_star = X\y;
            
            tic;
            %[~, k, residue, error] = LBFGS(w, f_lls, X, grad_lls, curr_l, tol, Wolfe, y, x_star);

            [~, k, residue, error] = LBFGS(w, f_lls, grad_lls, X, y, curr_l, tol, Wolfe, false, x_star);
            elapsed = toc;
            fprintf("Config: l= %d, lambda= %.1e, resid= %e, error= %.6e, iter= %d, time= %.2f\n", curr_l, curr_lambda, residue(end), error(end), k, elapsed)
            residues(ind, r, 1:length(residue)) = residue;
            errors(ind,r, 1:length(error)) = error;
            times(ind,r) = elapsed;
            iters(ind,r) = k;
            config(ind) = sprintf("L: %d, lam: %.2e", curr_l, curr_lambda);
            ind = ind + 1;
        end
    end
end
%residues = residues./runs;
%errors = errors./runs;
%times = times./runs;
%iters = iters./runs;
%disp(["l , lambda", "residues","" ,"errors", "iters", "times"]);
%disp([config, residues, errors, iters, times]);