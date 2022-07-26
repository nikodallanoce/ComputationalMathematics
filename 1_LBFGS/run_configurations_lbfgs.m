function [residues, errors, times, iters, config] = run_configurations_lbfgs(runs, l, lambda, tol, verbose)
addpath ../utilities;
% addpath ArmijoWolfeImplementations\;
tot_runs = length(lambda)*length(l);
residues = ones(tot_runs, runs);
errors = -ones(tot_runs, runs,1000);
times = zeros(tot_runs, runs);
iters = zeros(tot_runs, runs);
config = strings(tot_runs,1);
for r=1: runs
%     fprintf("Run %d \n \n", r);
    ind = 1;
    for i=1:length(l)
        curr_l=l(i);
        for j=1:length(lambda)
            curr_lambda = lambda(j);
            [X_hat, y_hat, w, w_star] = build_matrices("../datasets/ML-CUP21-TR.csv", curr_lambda);
            [f_lls, grad_lls] = build_lls(X_hat, y_hat);
            w = zeros(size(w));
            b = X_hat'*y_hat;
            tic;
             [~, k, error, resid] = LBFGS(w, sparse(X_hat),b, y_hat, l(i), tol, verbose, w_star);
%              LBFGS_risto(w, sparse(X_hat), b, y_hat, 20, 1e-14, true, w_star);
            elapsed = toc;
            fprintf("Config: l= %d, lambda= %.1e, resid= %e, error= %.6e, iter= %d, time= %.4e\n", curr_l, curr_lambda, resid, error(end), k, elapsed);
            residues(ind, r) = resid;
            errors(ind,r,1:length(error)) = error;
            times(ind,r) = elapsed;
            iters(ind,r) = k;
            config(ind) = sprintf("L: %d, lam: %.2e", curr_l, curr_lambda);
            ind = ind + 1;
        end
    end
end
rmpath ../utilities;
%residues = residues./runs;
%errors = errors./runs;
%times = times./runs;
%iters = iters./runs;
%disp(["l , lambda", "residues","" ,"errors", "iters", "times"]);
%disp([config, residues, errors, iters, times]);