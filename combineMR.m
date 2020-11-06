function [phi, Q] = combineMR(loc, basis, rho, derivative)
    [n, ndim] = size(loc);
    nlev = length(basis);

    nr = n;
    if derivative
        nr = ndim * n;
    end

    m = 0;
    for ilev = 1: nlev
        m = m + size(basis{ilev}.loc, 1);
    end

    iphi = zeros(1, nr*m);
    jphi = zeros(1, nr*m);
    phi = zeros(1, nr*m);

    iQ = zeros(1, m^2);
    jQ = zeros(1, m^2);
    Q = zeros(1, m^2);

    kphi = 1;
    kQ = 1;
    for ilev = 1: nlev
        [iphi0, jphi0, phi0] = regression(loc, basis{ilev}, derivative);
        k0 = length(phi0);
        iphi(kphi: kphi+k0-1) = iphi0;
        jphi(kphi: kphi+k0-1) = jphi0;
        phi(kphi: kphi+k0-1) = phi0;
        kphi = kphi + k0;

        [iQ0, jQ0, Q0] = precision(basis{ilev}, rho);
        k0 = length(Q0);
        iQ(kQ: kQ+k0-1) = iQ0;
        jQ(kQ: kQ+k0-1) = jQ0;
        Q(kQ: kQ+k0-1) = Q0;
        kQ = kQ + k0;
    end

    iphi = iphi(1: kphi-1);
    jphi = jphi(1: kphi-1);
    phi = phi(1: kphi-1);

    iQ = iQ(1: kQ-1);
    jQ = jQ(1: kQ-1);
    Q = Q(1: kQ-1);

    phi = sparse(iphi, jphi, phi, nr, m);
    Q = sparse(iQ, jQ, Q, m, m);
end