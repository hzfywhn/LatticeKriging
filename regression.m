function [ia, ja, a] = regression(obs, basis, derivative)
% calculate the regression matrix of one level
% obs: input, struct containing the location and direction cosine
% basis: input, struct containing the location and width
% derivative: input, bool indicating whether scalar or vector modeling is performed
% ia, ja, a: output, triplet indicating the indices and values of a matrix to be used in combineMR

% pair-wise distance between observations and basis functions
    d = pdist2(obs.loc, basis.loc) / basis.delta;
    d(d > 1) = NaN;

    if derivative
% for vector modeling, line-of-sight projection is performed (useful for curl-free fields)
        r1 = repmat(sum(obs.loc.*obs.dircos, 2), [1 size(basis.loc, 1)]);
        r2 = obs.dircos * basis.loc';
        phi = (1 - d).^5 .* (5*d + 1) * 56/3 / basis.delta^2 .* (r1 - r2);
    else
% for scalar modeling, Wendland function is used
        phi = (1 - d).^6 .* (35*d.^2 + 18*d + 3) / 3;
    end
    phi(isnan(phi)) = 0;

    [ia, ja, a] = find(sparse(phi));
end