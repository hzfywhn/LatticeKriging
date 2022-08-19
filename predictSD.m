function sd = predictSD(Z1, phi1, lambda, W, Z, Q, phi, M, rhoMLE)
    y1 = phi * (Q \ phi1');
    ZMZ = Z' * (M \ Z);
    d1 = ZMZ \ Z' * (M \ y1);
    r1 = y1 - Z*d1;
    c1 = Q \ phi' * (M \ r1);
    residual = y1 - Z*d1 - phi*c1;
    joint = diag(Z1 * (ZMZ \ Z1')) - 2 * diag(Z1 * d1);

    weight = chol(Q)' \ phi1';
%     normweight = diag(weight' * weight);
    normweight = sum(weight.^2, 1);
    normweight(normweight == 0) = 1;
    marginal = normweight' - diag(y1' * W * residual) / lambda;

    sd = sqrt(rhoMLE * abs(joint + marginal));
end