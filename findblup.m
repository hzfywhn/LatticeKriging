function [x, fval, exitflag, output] = findblup(y, W, Z, Q, phi, logxmin, logxmax, options)
% find the best unbiased linear prediction of the model by maximizing likelihood over the parameter space
% y, W, Z, Q, phi: input, all can be found on Nychka et al. (2015)
% logxmin, logxmax, options: input, fminbnd parameters
% x: output, best estimate of log(lambda)
% fval: output, minus maximum likelihood
% exitflag, output: output, fminbnd outputs

    function likelihood = LK(lambda)
        [~, ~, ~, likelihood, ~] = kriging(exp(lambda), y, W, Z, Q, phi);
% maximizing a positive function is equivalent to minimizing a negative function
        likelihood = -likelihood;
    end
    [x, fval, exitflag, output] = fminbnd(@LK, logxmin, logxmax, options);
end