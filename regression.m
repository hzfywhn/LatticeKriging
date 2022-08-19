function [ia, ja, a] = regression(obs, basis, derivative)
    d = pdist2(obs.loc, basis.loc) / basis.delta;
    d(d > 1) = NaN;
    if derivative
        r1 = repmat(sum(obs.loc.*obs.dircos, 2), [1 size(basis.loc, 1)]);
        r2 = obs.dircos * basis.loc';
        phi = (1 - d).^5 .* (5*d + 1) * 56/3 / basis.delta^2 .* (r1 - r2);
    else
        phi = (1 - d).^6 .* (35*d.^2 + 18*d + 3) / 3;
    end
    phi(isnan(phi)) = 0;

    [ia, ja, a] = find(sparse(phi));
end