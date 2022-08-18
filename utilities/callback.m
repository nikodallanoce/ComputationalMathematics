function callback(x, k, w_star, X_hat, y_hat)
    global metrics;
    metrics.rel_errors(k) = norm(x - w_star)/norm(w_star);
    metrics.residual(k) = norm(X_hat*x - y_hat)/norm(y_hat);
end

