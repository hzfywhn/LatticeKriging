function [ia, ja, a] = precision(basis, rho)
    [ia, ja, a] = SAR(basis);
    m = length(a);
    B = sparse(ia, ja, a, m, m);
    Q = B' * B / rho;
    [ia, ja, a] = find(Q);
end