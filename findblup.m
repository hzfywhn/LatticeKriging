function [x, fval, exitflag, output] = findblup(y, W, Z, Q, phi, xmin, xmax, options)
    function likelihood = LK(lambda)
        [~, ~, ~, likelihood, ~] = kriging(lambda, y, W, Z, Q, phi);
        likelihood = -likelihood;
    end
    [x, fval, exitflag, output] = fminbnd(@LK, xmin, xmax, options);
end