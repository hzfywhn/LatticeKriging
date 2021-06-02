function [d, c, rhoMLE, likelihood, M] = kriging(lambda, y, W, Z, Q, phi)
    n = length(y);

    M = full(phi * (Q \ phi')) + lambda * inv(W);

    d = (Z' * (M \ Z)) \ Z' * (M \ y);
    r = y - Z*d;
    c = full(Q \ phi') * (M \ r);
    rhoMLE = r' * (M \ r) / n;

    [Mc, flag] = chol(M);
    assert(flag == 0)
    detM_2 = sum(log(diag(Mc)));
    likelihood = -n/2 - n/2*log(rhoMLE) - detM_2 - n/2*log(2*pi);
end