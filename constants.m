function [y, W, Z, phi, Q] = constants(obs, basis, normalization, rho, derivative)
    y = obs.val(:);
    W = errorCov(obs.err(:));

    Z = covariate(obs.loc);
    if derivative
        Z = covariate(repmat(obs.loc, 1, size(obs.loc, 2)));
    end

    [phi, Q] = combineMR(obs.loc, basis, rho, derivative);

    if normalization
        [Qc, flag] = chol(Q);
        assert(flag == 0)
        normweight = rho * sum((Qc' \ phi').^2, 1);
        assert(all(normweight ~= 0))
        ind = 1: length(normweight);
        phi = sparse(ind, ind, 1./sqrt(normweight)) * phi;
    end
end