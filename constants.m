function [y, W, Z, Q, phi] = constants(obs, basis, normalization, rho, derivative)
    y = obs.val(:);
    ind = 1: numel(obs.err);
    W = sparse(ind, ind, 1./obs.err(:).^2);

    [n, ndim] = size(obs.loc);
    Z = [ones(n, 1) obs.loc];
    if derivative
        Z = [ones(n*ndim, 1) repmat(obs.loc, ndim, 1)];
    end

    [Q, phi] = combineMR(obs.loc, basis, normalization, rho, derivative);
end