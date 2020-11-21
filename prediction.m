function [m, sd] = prediction(loc, basis, normalization, rho, lambda, Z, Q, phi, M, d, c, rhoMLE)
    Z1 = [ones(size(loc, 1), 1) loc];
    obs.loc = loc;
    [~, phi1] = combineMR(obs, basis, normalization, rho, false);

    [Qc, flag] = chol(Q);
    assert(flag == 0)
    normweight = sum((Qc' \ phi1').^2, 1);
    assert(all(normweight ~= 0))

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