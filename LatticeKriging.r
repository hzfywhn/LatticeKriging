# a simplified rewrite of LatticeKrig package on Cartesian geometry

library(package = "spam")

SAR <- function(basis) {
    # spatial autogression
    m <- nrow(basis$loc)
    m0 <- ncol(basis$connect)

    ia <- array(dim = m*m0)
    ja <- array(dim = m*m0)
    a <- array(dim = m*m0)

    k <- 1
    for (j in 1: m) {
        ia[k] <- j
        ja[k] <- j
        a[k] <- basis$centerweight
        k <- k + 1

        for (j0 in 1: m0) {
            js <- basis$connect[j, j0]
            if (!is.na(js)) {
                ia[k] <- j
                ja[k] <- js
                a[k] <- -1
                k <- k + 1
            }
        }
    }

    ia <- ia[1: (k-1)]
    ja <- ja[1: (k-1)]
    a <- a[1: (k-1)]

    return (list(i = ia, j = ja, values = a))
}

regression <- function(obs, basis, derivative) {
    d <- rdist::cdist(obs$loc, basis$loc) / basis$delta
    d[d > 1] <- NA
    if (derivative) {
        r1 <- replicate(n = nrow(basis$loc), expr = rowSums(obs$loc * obs$azim))
        r2 <- obs$azim[, 1] %*% t(basis$loc[, 1])
        for (idim in 2: ncol(obs$loc)) r2 <- r2 + obs$azim[, idim] %*% t(basis$loc[, idim])
        phi <- (1 - d)^5 * (5*d + 1) * 56/3 / basis$delta^2 * (r1 - r2)
    } else {
        phi <- (1 - d)^6 * (35*d^2 + 18*d + 3) / 3
    }
    phi[is.na(phi)] <- 0

    return (triplet(as.spam(phi), tri = TRUE))
}

combineMR <- function(obs, basis, normalization, rho, derivative) {
    n <- nrow(obs$loc)
    nlev <- length(basis)

    m0 <- array(dim = nlev)
    for (ilev in 1: nlev) m0[ilev] <- nrow(basis[[ilev]]$loc)
    m1 <- cumsum(c(0, m0))
    m <- m1[nlev+1]

    iB <- array(dim = m^2)
    jB <- array(dim = m^2)
    B <- array(dim = m^2)

    iphi <- array(dim = n*m)
    jphi <- array(dim = n*m)
    phi <- array(dim = n*m)

    kB <- 1
    kphi <- 1
    for (ilev in 1: nlev) {
        B0 <- SAR(basis[[ilev]])
        phi0 <- regression(obs, basis[[ilev]], derivative)

        if (normalization) {
            # don't understand how it is related to Sec 2.6
            B1 <- spam(x = B0, nrow = m0[ilev], ncol = m0[ilev])
            Q1 <- t(B1) %*% B1
            phi1 <- spam(x = phi0, nrow = n, ncol = m0[ilev])
            weight <- forwardsolve(chol(Q1), t(phi1))
            normweight <- diag(t(weight) %*% weight)
            normweight[normweight == 0] <- 1
            phi1 <- diag.spam(x = 1/sqrt(normweight)) %*% phi1
            phi0 <- triplet(phi1, tri = TRUE)
        }

        k0 <- length(B0$values)
        iB[kB: (kB+k0-1)] <- B0$i + m1[ilev]
        jB[kB: (kB+k0-1)] <- B0$j + m1[ilev]
        # this is different from Eq 11, there is a scaling factor in front
        B[kB: (kB+k0-1)] <- B0$values
        kB <- kB + k0

        k0 <- length(phi0$values)
        iphi[kphi: (kphi+k0-1)] <- phi0$i
        jphi[kphi: (kphi+k0-1)] <- phi0$j + m1[ilev]
        # scaling factor of precision matrix moved here, not clear why
        phi[kphi: (kphi+k0-1)] <- phi0$values * sqrt(basis[[ilev]]$alpha)
        kphi <- kphi + k0
    }

    iB <- iB[1: (kB-1)]
    jB <- jB[1: (kB-1)]
    B <- B[1: (kB-1)]

    iphi <- iphi[1: (kphi-1)]
    jphi <- jphi[1: (kphi-1)]
    # scaling factor of precision matrix moved here, not clear why
    phi <- phi[1: (kphi-1)] * sqrt(rho)

    B <- spam(x = list(i = iB, j = jB, values = B), nrow = m, ncol = m)
    Q <- t(B) %*% B
    phi <- spam(x = list(i = iphi, j = jphi, values = phi), nrow = n, ncol = m)

    return (list(Q = Q, phi = phi))
}

kriging <- function(lambda, y, W, Z, Q, phi) {
    n <- length(y)

    # auxiliary matrix M_lambda
    M <- phi %*% solve(Q, t(phi)) + lambda * chol2inv(chol(W))

    # here are the most expansive computations taken place
    d <- solve(t(Z) %*% solve(M, Z), t(Z)) %*% solve(M, y)
    r <- y - Z %*% d
    c <- solve(Q, t(phi)) %*% solve(M, r)
    rhoMLE <- t(r) %*% solve(M, r) / n
    likelihood <- -n/2 - n/2*log(rhoMLE) - sum(log(diag(chol(M))))
    # the last term +n/2*log(pi) in Eq 7 was wrong, corrected here
    likelihood <- likelihood - n/2*log(2*pi)

    return (list(d = drop(d), c = drop(c), rhoMLE = drop(rhoMLE), likelihood = drop(likelihood), M = M))
}

predictMean <- function(Z1, phi1, d, c) {
    return (drop(Z1 %*% d + phi1 %*% c))
}

predictSD <- function(Z1, phi1, lambda, W, Z, Q, phi, M, rhoMLE) {
    # standard deviation prediction is confusing with no reference found, use with caution

    y1 <- phi %*% solve(Q, t(phi1))
    ZMZ <- t(Z) %*% solve(M, Z)
    d1 <- solve(ZMZ, t(Z)) %*% solve(M, y1)
    r1 <- y1 - Z %*% d1
    c1 <- solve(Q, t(phi)) %*% solve(M, r1)
    residual <- y1 - Z %*% d1 - phi %*% c1
    joint <- diag(Z1 %*% solve(ZMZ, t(Z1))) - 2 * diag(Z1 %*% d1)

    weight <- forwardsolve(chol(Q), t(phi1))
    normweight <- diag(t(weight) %*% weight)
    normweight[normweight == 0] <- 1
    marginal <- normweight - diag(t(y1) %*% W %*% residual) / lambda

    sd <- sqrt(rhoMLE * abs(joint + marginal))
    return (sd)
}

# codes below are for testing
if (FALSE) {
    # test domain
    x1 <- -4
    x2 <- 4
    y1 <- -2
    y2 <- 2

    # basis set
    delta <- 0.2
    x0 <- seq(from = x1, to = x2, by = delta)
    y0 <- seq(from = y1, to = y2, by = delta)
    nx <- length(x0)
    ny <- length(y0)
    loc <- array(dim = c(nx*ny, 2))
    con <- array(dim = c(nx*ny, 4))
    for (j in 1: ny) {
        for (i in 1: nx) {
            idx <- (j-1)*nx + i
            loc[idx, 2] <- y0[j]
            loc[idx, 1] <- x0[i]
            if (j >= 2) con[idx, 1] <- (j-2)*nx + i
            if (i >= 2) con[idx, 2] <- (j-1)*nx + i-1
            if (j <= ny-1) con[idx, 3] <- j*nx + i
            if (i <= nx-1) con[idx, 4] <- (j-1)*nx + i+1
        }
    }
    basis <- list(list(loc = loc, connect = con, centerweight = 4.01, delta = delta*2.5, alpha = 1))

    normalization <- FALSE
    rho <- 1

    # observation
    n <- 1200
    x <- runif(n, min = x1, max = x2)
    y <- runif(n, min = y1, max = y2)
    t <- runif(n, min = 0, max = 2*pi)

    vx <- 2*((x-1)/(1+(x-1)^2+y^2)^2 - (x+1)/(1+(x+1)^2+y^2)^2)
    vy <- 2*(y/(1+(x-1)^2+y^2)^2 - y/(1+(x+1)^2+y^2)^2)

    yfit <- vx*cos(t) + vy*sin(t)
    W <- diag.spam(x = 1, nrow = n, ncol = n)
    Z <- cos(t)

    MR <- combineMR(list(loc = cbind(x, y), azim = cbind(cos(t), sin(t))), basis, normalization, rho, TRUE)
    # interval needs to be adjusted for specific cases
    lambda <- exp(optimize(
        function(l) kriging(exp(l), yfit, W, Z, MR$Q, MR$phi)$likelihood,
        interval = c(4, 9), maximum = TRUE)$maximum)
    fit <- kriging(lambda, yfit, W, Z, MR$Q, MR$phi)

    delta <- 0.1
    x0 <- seq(from = x1, to = x2, by = delta)
    y0 <- seq(from = y1, to = y2, by = delta)
    loc <- expand.grid(x0, y0)
    xo <- loc[, 1]
    yo <- loc[, 2]
    Z1 <- as.matrix(xo)
    MR1 <- combineMR(list(loc = cbind(xo, yo)), basis, normalization, rho, FALSE)
    m <- predictMean(Z1, MR1$phi, fit$d, fit$c)
    sd <- predictSD(Z1, MR1$phi, lambda, W, Z, MR$Q, MR$phi, fit$M, fit$rhoMLE)
}