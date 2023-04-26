function [d, c, rhoMLE, likelihood, M] = kriging(lambda, y, W, Z, Q, phi)
% lambda, y, W, Z, Q, phi: input, all can be found on Nychka et al. (2015)
% d: output, estimate of the prior bias
% c: output, estimate of the basis function coefficients
% rhoMLE: output, estimate of the marginal variance
% likelihood: output, likelihood for the current parameter, to be used in findblup
% M: output, auxiliary matrix

    n = length(y);

    M = phi * (Q \ phi') + lambda * sparse(inv(W));

    d = (Z' * (M \ Z)) \ Z' * (M \ y);
    r = y - Z*d;
    c = Q \ phi' * (M \ r);
    rhoMLE = r' * (M \ r) / n;

    likelihood = -n/2 - n/2*log(rhoMLE) - sum(log(diag(chol(M)))) - n/2*log(2*pi);
end