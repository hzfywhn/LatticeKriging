function W = errorCov(err)
    ind = 1: length(err);
    W = sparse(ind, ind, 1./err.^2);
end