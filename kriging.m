function [d, c, rhoMLE, likelihood, M] = kriging(lambda, y, W, Z, Q, phi)
    n = length(y);

    M = phi * (Q \ phi') + lambda * sparse(inv(W));

    d = (Z' * (M \ Z)) \ Z' * (M \ y);
    r = y - Z*d;
    c = Q \ phi' * (M \ r);
    rhoMLE = r' * (M \ r) / n;

    likelihood = -n/2 - n/2*log(rhoMLE) - sum(log(diag(chol(M)))) - n/2*log(2*pi);
end