clear;

lambdas = [1e4, 1e2, 1, 1e-2, 1e-4];
etas = [0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7];
alphas = 0.1:0.1:0.9;
verbose = 0;
configs = zeros(length(lambdas),4);
for l = 1:length(lambdas)
    lambda = lambdas(l);
    min_err = 1;
    final_eta = 0;
    final_alpha = 0;
    final_k = 0;
    for i = 1:length(etas)
        % disp(etas(i));
        for j = 1:length(alphas)
            eta = etas(i);
            alpha = alphas(j);
            [x, k, errors, residuals, abs_err] = run(lambda, eta, alpha, verbose);
            % disp(abs_err);
            if abs_err <= min_err
                final_eta = eta;
                final_alpha = alpha;
                min_err = abs_err;
                final_k = k;
            end
        end
    end
    fprintf('Lambda=%1.4e eta=%1.7e alpha=%1.1e norm(w*-w)=%1.16e k=%1.5e\n',lambda, final_eta, final_alpha, min_err, k);
    configs(l,1) = final_k;
    configs(l,2) = abs_err;
    configs(l,3) = final_eta;
    configs(l,4) = final_alpha;
end

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
