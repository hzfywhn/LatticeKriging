function [y, W, Z, Q, phi] = constants(obs, basis, normalization, rho, derivative)
    y = obs.val;
    ind = 1: numel(obs.err);
    W = sparse(ind, ind, 1./obs.err.^2);
    Z = [ones(size(obs.loc, 1), 1) obs.loc];
    [Q, phi] = combineMR(obs, basis, normalization, rho, derivative);
end