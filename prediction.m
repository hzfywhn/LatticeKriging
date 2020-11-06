function [m, sd] = prediction(loc, basis, normalization, rho, lambda, Z, phi, Q, M, d, c, rhoMLE)
    Z1 = covariate(loc);
    [phi1, ~] = combineMR(loc, basis, rho, false);

    [Qc, flag] = chol(Q);
    assert(flag == 0)
    normweight = rho * sum((Qc' \ phi1').^2, 1);
    assert(all(normweight ~= 0))
    if normalization
        ind = 1: length(normweight);
        phi1 = sparse(ind, ind, 1./sqrt(normweight)) * phi1;
    end

    m = Z1*d + phi1*c;

    y1 = phi * (Q \ phi1');
    ZMZ = Z' * (M \ Z);
    d1 = ZMZ \ Z' * (M \ y1);
    r1 = y1 - Z*d1;
    c1 = Q \ phi' * (M \ r1);
    residual = y1 - Z*d1 - phi*c1;
    joint = sum(Z1' .* (ZMZ \ Z1'), 1) - 2 * sum(Z1' .* d1, 1);
    marginal = normweight - sum(y1 .* residual, 1) / lambda;
    sd = sqrt(rhoMLE * abs(joint + marginal));
end