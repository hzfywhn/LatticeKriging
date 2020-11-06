function Z = covariate(loc)
    Z = [ones(size(loc, 1), 1) loc];
end