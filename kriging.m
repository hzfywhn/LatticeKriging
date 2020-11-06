function [d, c, rhoMLE, likelihood, M] = kriging(lambda, y, W, Z, phi, Q)
    n = length(y);

    M = phi * (Q \ phi') + lambda * inv(W);

    d = Z' * (M \ Z) \ Z' * (M \ y);
    r = y - Z*d;
    c = Q \ phi' * (M \ r);
    rhoMLE = r' * (M \ r) / n;

    [Mc, flag, ~] = chol(M);
    assert(flag == 0)
    likelihood = -n/2 - n/2*log(rhoMLE) - sum(log(diag(Mc))) - n/2*log(2*pi);
end