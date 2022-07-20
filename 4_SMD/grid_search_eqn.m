clear;

lambdas = [1e4, 1e2, 1, 1e-2, 1e-4];
betas = 0.0:0.05:0.9;
verbose = false;
configs = zeros(length(lambdas),3);

for l = 1:length(lambdas)
    lambda = lambdas(l);
    min_err = 1;
    min_k = inf;
    final_alpha = 0;
    final_k = 0;

    for j = 1:length(betas)
        alpha = betas(j);
        [x, k, errors, residuals, abs_err] = run_eqn(lambda, alpha, verbose, 3e3);
        % disp(abs_err);
        if k <= min_k
            final_alpha = alpha;
            min_err = abs_err;
            min_k = k;
            final_k = k;
        end
        disp(alpha);
    end
    
    fprintf('Lambda=%1.4e alpha=%1.1e norm(w*-w)=%1.16e k=%1.5e\n', lambda, final_alpha, min_err, k);
    configs(l,1) = final_k;
    configs(l,2) = abs_err;
    configs(l,3) = final_alpha;
end
%{
%refinement
disp("Refinement of the grid on eta values.")
for i =1:length(lambdas)
    min_err = 1;
    final_eta = 0;
    final_alpha = 0;
    final_k = 0;
    
    eta = configs(i,3);
    alpha = configs(i,4);
    lambda = lambdas(i);
    mean = eta/2;
    max_eta = eta+mean;
    min_eta = eta-mean;
    step_size = eta/10;
    eta_refinement = min_eta:step_size:max_eta;
    for j = 1:length(eta_refinement)
        new_eta = eta_refinement(j);
        [x, k, errors, residuals, abs_err] = run(lambda, new_eta, alpha, verbose);
        if abs_err <= min_err
                final_eta = new_eta;
                final_alpha = alpha;
                min_err = abs_err;
                final_k = k;
        end
    end
    fprintf('Lambda=%1.4e eta=%1.7e alpha=%1.1e norm(w*-w)=%1.16e k=%1.5e\n',lambda, final_eta, final_alpha, min_err, k);
end
%}
