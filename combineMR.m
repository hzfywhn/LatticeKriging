function [Q, phi] = combineMR(obs, basis, normalization, rho, derivative)
% combine multi-resolution precision matrices and regression matrices into one
% obs: input, struct containing the location and direction cosine
% basis: input, struct containing the location and connectivity matrix of basis functions
% normalization: input, bool indicating whether the regression matrix needs normalization
%   turning it on might improve the performance near the edge, but modeling with a larger area (sponge) is suggested
% rho: input, real representing the initial guess of the marginal variance, it will be re-estimated in kriging
% derivative: input, bool indicating whether scalar or vector modeling is performed
% Q: output, full precision matrix including all levels
% phi: output, full regression matrix including all levels

    n = size(obs.loc, 1);
    nlev = length(basis);

    m0 = zeros(1, nlev);
    for ilev = 1: nlev
        m0(ilev) = size(basis{ilev}.loc, 1);
    end
    m1 = cumsum([0 m0]);
    m = m1(nlev+1);

    iB = zeros(1, m^2);
    jB = zeros(1, m^2);
    B = zeros(1, m^2);

    iphi = zeros(1, n*m);
    jphi = zeros(1, n*m);
    phi = zeros(1, n*m);

    kB = 1;
    kphi = 1;
    for ilev = 1: nlev
        [iB0, jB0, B0] = SAR(basis{ilev});
        [iphi0, jphi0, phi0] = regression(obs, basis{ilev}, derivative);

% calculate the normalization weights based on the original LatticeKrig package
        if normalization
            B1 = sparse(iB0, jB0, B0, m0(ilev), m0(ilev));
            Q1 = B1' * B1;
            phi1 = sparse(iphi0, jphi0, phi0, n, m0(ilev));
            [Qc, flag] = chol(Q1);
            assert(flag == 0)
            weight = Qc' \ phi1';
            normweight = sum(weight.^2, 1);
            normweight(normweight == 0) = 1;
            phi1 = spdiags(1./sqrt(normweight), 0, n, n) * phi1;
            [iphi0, jphi0, phi0] = find(phi1);
        end

% stack SAR matrices diagonally
        k0 = length(B0);
        iB(kB: kB+k0-1) = iB0 + m1(ilev);
        jB(kB: kB+k0-1) = jB0 + m1(ilev);
        B(kB: kB+k0-1) = B0;
        kB = kB + k0;

% stack regression matrices horizontally
        k0 = length(phi0);
        iphi(kphi: kphi+k0-1) = iphi0;
        jphi(kphi: kphi+k0-1) = jphi0 + m1(ilev);
        phi(kphi: kphi+k0-1) = phi0 * sqrt(basis{ilev}.alpha);
        kphi = kphi + k0;
    end

    iB = iB(1: kB-1);
    jB = jB(1: kB-1);
    B = B(1: kB-1);

    iphi = iphi(1: kphi-1);
    jphi = jphi(1: kphi-1);
    phi = phi(1: kphi-1) * sqrt(rho);

    B = sparse(iB, jB, B, m, m);
    Q = B' * B;
    phi = sparse(iphi, jphi, phi, n, m);
end