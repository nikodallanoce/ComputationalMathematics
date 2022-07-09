clear;

lambda = [1e4, 1e2, 1, 1e-2, 1e-4];
l = [5, 10, 15, 20];

[residues, errors, times, iters, config] = run_configurations_lbfgs(3, l, lambda, 1e-12, true);
%csvwrite('wolfe_errors_config_cin.csv',errors)