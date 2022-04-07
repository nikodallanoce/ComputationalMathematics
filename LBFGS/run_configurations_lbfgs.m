function run_configurations_lbfgs(runs, l, lambda, tol, D, y)
tot_runs = length(lambda)*length(l);
residues = zeros(tot_runs,1);
errors = zeros(tot_runs,1);
times = zeros(tot_runs,1);
iters = zeros(tot_runs, 1);
config = strings(tot_runs, 1);
ind = 1;
[m0, ~] = size(D);
% Compute LBFGS
X = [D'; 1.*eye(m0)];
[~, n] = size(X);
for r=1: runs
    fprintf("Run %d \n \n", r);
    w = randn(n, 1);
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

            w_exact = X\y;
            tic;
            [w_our, k] = LBFGS(w, f_lls, X, grad_lls, curr_l, tol, false);
            elapsed = toc;
            residue = norm(X*w_our-y);
            error = norm(w_our - w_exact)/norm(w_exact);
            fprintf("Config: l= %d, lambda= %.1e, resid= %e, error= %.6e, iter= %d, time= %.2f\n", curr_l, curr_lambda, residue, error, k, elapsed)
            residues(ind) = residues(ind) + residue;
            errors(ind) = errors(ind) + error;
            times(ind) = times(ind) + elapsed;
            iters(ind) = iters(ind) + k;
            config(ind) = sprintf("%d %.2e",curr_l, curr_lambda);
            ind = ind +1;
        end
    end
    ind = 1;
end
residues = residues./runs;
errors = errors./runs;
times = times./runs;
iters = iters./runs;
disp(["l , lambda", "residues","" ,"errors", "iters", "times"]);
disp([config, residues, errors, iters, times]);