function [y, W, Z, Q, phi] = constants(obs, basis, normalization, rho, derivative)
    y = obs.val(:);
    ind = 1: length(obs.err);
    W = sparse(ind, ind, 1./obs.err(:).^2);

    [n, ndim] = size(obs.loc);
    one = ones(n, 1);
    Z = [one obs.loc];
    if derivative
        Z = [one repmat(obs.loc, ndim, 1)];
    end

    [Q, phi] = combineMR(obs.loc, basis, normalization, rho, derivative);
end