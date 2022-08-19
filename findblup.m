function [x, fval, exitflag, output] = findblup(y, W, Z, Q, phi, logxmin, logxmax, options)
    function likelihood = LK(lambda)
        [~, ~, ~, likelihood, ~] = kriging(exp(lambda), y, W, Z, Q, phi);
        likelihood = -likelihood;
    end
    [x, fval, exitflag, output] = fminbnd(@LK, logxmin, logxmax, options);
end