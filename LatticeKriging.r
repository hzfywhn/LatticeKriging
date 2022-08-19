# a simplified rewrite of LatticeKrig package on Cartesian geometry

library(package = "spam")
library(package = "spam64")

SAR <- function(connect, centerweight) {
# spatial auto-regression (B matrix in triplet form)
    m <- nrow(connect)
    m0 <- ncol(connect)

    ia <- array(dim = m * m0)
    ja <- array(dim = m * m0)
    a <- array(dim = m * m0)

    k <- 1
    for (j in 1: m) {
        ia[k] <- j
        ja[k] <- j
        a[k] <- centerweight
        k <- k + 1

        for (j0 in 1: m0) {
            js <- connect[j, j0]
            if (!is.na(js)) {
                ia[k] <- j
                ja[k] <- js
                a[k] <- -1
                k <- k + 1
            }
        }
    }

    ia <- ia[1: (k - 1)]
    ja <- ja[1: (k - 1)]
    a <- a[1: (k - 1)]

    return(list(i = ia, j = ja, values = a))
}

precision <- function(connect, centerweight) {
# combine all levels to form the full precision matrix
    nlev <- length(connect)

    m0 <- array(dim = nlev)
    for (ilev in 1: nlev) m0[ilev] <- nrow(connect[[ilev]])
    m1 <- cumsum(c(0, m0))
    m <- m1[nlev + 1]

    iB <- array(dim = m^2)
    jB <- array(dim = m^2)
    B <- array(dim = m^2)

    k <- 1
    for (ilev in 1: nlev) {
        B0 <- SAR(connect[[ilev]], centerweight)

        k0 <- length(B0$values)
        iB[k: (k + k0 - 1)] <- B0$i + m1[ilev]
        jB[k: (k + k0 - 1)] <- B0$j + m1[ilev]
        B[k: (k + k0 - 1)] <- B0$values
        k <- k + k0
    }

    iB <- iB[1: (k - 1)]
    jB <- jB[1: (k - 1)]
    B <- B[1: (k - 1)]

    B <- spam(list(i = iB, j = jB, values = B), nrow = m, ncol = m)
    Q <- t(B) %*% B

    return(Q)
}

basis <- function(x, dircos, x0, width) {
# pair-wise distance between observations and basis functions in triplet form
    d <- rdist::cdist(x, x0) / width
    d[d > 1] <- NA
    if (is.null(dircos)) {
        phi <- (1 - d)^6 * (35 * d^2 + 18 * d + 3) / 3
    } else {
        r1 <- replicate(n = nrow(x0), expr = rowSums(x * dircos))
        r2 <- dircos %*% t(x0)
        phi <- (1 - d)^5 * (5 * d + 1) * 56 / 3 / width^2 * (r1 - r2)
    }
    phi[is.na(phi)] <- 0

    return(triplet(phi, tri = TRUE))
}

regression <- function(x, dircos, x0, width, alpha, rho) {
# combine all levels to form the full regression matrix
    n <- nrow(x)
    nlev <- length(x0)

    m0 <- array(dim = nlev)
    for (ilev in 1: nlev) m0[ilev] <- nrow(x0[[ilev]])
    m1 <- cumsum(c(0, m0))
    m <- m1[nlev + 1]

    iphi <- array(dim = n * m)
    jphi <- array(dim = n * m)
    phi <- array(dim = n * m)

    k <- 1
    for (ilev in 1: nlev) {
        phi0 <- basis(x, dircos, x0[[ilev]], width[ilev])

        k0 <- length(phi0$values)
        iphi[k: (k + k0 - 1)] <- phi0$i
        jphi[k: (k + k0 - 1)] <- phi0$j + m1[ilev]
        phi[k: (k + k0 - 1)] <- phi0$values * sqrt(alpha[ilev])
        k <- k + k0
    }

    iphi <- iphi[1: (k - 1)]
    jphi <- jphi[1: (k - 1)]
    phi <- phi[1: (k - 1)] * sqrt(rho)

    phi <- spam(list(i = iphi, j = jphi, values = phi), nrow = n, ncol = m)

    return(phi)
}

kriging <- function(wy, wU, wX, Q, lambda, logdetcov, Gc) {
# calculate coefficients and likelihood
    n <- nrow(wy)
    nreps <- ncol(wy)

    Gc <- chol(t(wX) %*% wX + lambda * Q, Rstruct = Gc)

    A <- t(wU) %*% (wU - wX %*% backsolve(Gc, forwardsolve(Gc, t(wX) %*% wU))) / lambda
    b <- t(wU) %*% (wy - wX %*% backsolve(Gc, forwardsolve(Gc, t(wX) %*% wy))) / lambda

    omega <- solve(A)
    d <- omega %*% b
    r <- wy - wU %*% d

    c <- forwardsolve(Gc, t(wX) %*% r)
    rhoMLE <- 1/lambda * (colSums(as.matrix(r^2)) - colSums(as.matrix(c^2))) / n
    c <- backsolve(Gc, c)

    like <- nreps * (logdetcov - n/2*log(mean(rhoMLE)) - sum(log(diag(Gc))))

    return(list(d = drop(d), c = c, rhoMLE = rhoMLE, like = like, Gc = Gc, omega = omega))
}

LatticeKriging <- function(x, dircos, y, w, Z,
    x0, width, alpha, connect, centerweight, rho,
    xnew, Znew, pred_m, pred_se) {
# find the best estimate of lambda and predict
    phi <- regression(x, dircos, x0, width, alpha, rho)
    Q <- precision(connect, centerweight)

    n <- nrow(phi)
    m <- ncol(phi)
    wy <- as.matrix(sqrt(w) * y)
    wU <- as.matrix(sqrt(w) * Z)
    wX <- diag(sqrt(w)) %*% phi

    Gc <- NULL
    Qc <- chol(Q)
    logdetcov <- -n/2 - n/2*log(2*pi) + sum(log(diag(Qc))) + sum(log(w))/2

    logl <- optimize(function(l) {
        like <- kriging(wy, wU, wX, Q, exp(l), logdetcov-(n-m)/2*l, Gc)
        Gc <<- like$Gc
        return(like$like)
    }, interval = c(-10, 10), maximum = TRUE, tol = 1e-4)$maximum

    lambda <- exp(logl)
    coef <- kriging(wy, wU, wX, Q, lambda, logdetcov-(n-m)/2*log(lambda), Gc)
    res <- list(lambda = lambda, rhoMLE = coef$rhoMLE, like = coef$like, d = coef$d, c = coef$c)

    Xnew <- regression(xnew, NULL, x0, width, alpha, rho)
    Unew <- as.matrix(Znew)

    if (pred_m) {
        mn <- Unew %*% coef$d + Xnew %*% coef$c
        res <- append(res, list(m = drop(mn)))
    }

    if (pred_se) {
        Anew <- forwardsolve(Qc, t(Xnew))
        wk0 <- wX %*% backsolve(Qc, Anew)
        coefnew <- kriging(wk0, wU, wX, Q, lambda, logdetcov-(n-m)/2*log(lambda), Gc)
        cmkrig <- (wk0 - wU %*% coefnew$d - wX %*% coefnew$c) / lambda
        variance <- coef$rhoMLE * (colSums(t(Unew) * (coef$omega %*% t(Unew))) - 2 * colSums(t(Unew) * coefnew$d))
        marginal <- coef$rhoMLE * (colSums(Anew^2) - colSums(wk0 * cmkrig))
        se <- sqrt(variance + marginal)
        res <- append(res, list(se = drop(se)))
    }

    return(res)
}

# codes below are for testing
if (FALSE) {
    # test domain
    x1 <- -4
    x2 <- 4
    y1 <- -2
    y2 <- 2

    # observation
    delta <- 0.05
    loc <- expand.grid(seq(from = x1, to = x2, by = delta),
        seq(from = y1, to = y2, by = delta))
    xi <- loc[, 1]
    yi <- loc[, 2]

    vx <- 2*((xi-1)/(1+(xi-1)^2+yi^2)^2 - (xi+1)/(1+(xi+1)^2+yi^2)^2)
    vy <- 2*(yi/(1+(xi-1)^2+yi^2)^2 - yi/(1+(xi+1)^2+yi^2)^2)

    # basis function
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
            loc[idx, 1] <- x0[i]
            loc[idx, 2] <- y0[j]
            if (j >= 2) con[idx, 1] <- (j-2)*nx + i
            if (i >= 2) con[idx, 2] <- (j-1)*nx + i-1
            if (j <= ny-1) con[idx, 3] <- j*nx + i
            if (i <= nx-1) con[idx, 4] <- (j-1)*nx + i+1
        }
    }
    xb <- list(loc)
    width <- delta * 2.5
    alpha <- 1
    connect <- list(con)
    centerweight <- 4.01

    delta <- 0.1
    loc <- expand.grid(seq(from = x1, to = x2, by = delta),
        seq(from = y1, to = y2, by = delta))
    xo <- loc[, 1]
    yo <- loc[, 2]
    vo <- 1/(1+(xo-1)^2+yo^2) - 1/(1+(xo+1)^2+yo^2)

    # dircos <- NULL
    # vi <- 1/(1+(xi-1)^2+yi^2) - 1/(1+(xi+1)^2+yi^2)
    # Z <- cbind(1, xi, yi)
    # Znew <- cbind(1, xo, yo)
    t <- seq(from = 0, to = 2*pi, length.out = length(xi))
    dircos <- cbind(cos(t), sin(t))
    vi <- vx*cos(t) + vy*sin(t)
    Z <- -cos(t)
    Znew <- xo

    w <- rep(1, times = length(vi))

    rho <- 1
    lk <- LatticeKriging(cbind(xi, yi), dircos, vi, w, Z,
        xb, width, alpha, connect, centerweight, rho,
        cbind(xo, yo), Znew, TRUE, TRUE)

    # for comparison
    # lk0 <- LatticeKrig::LatticeKrig(cbind(xi, yi), vi, nlevel = 1, NC = nx, NC.buffer = 0, normalize = FALSE)
    # m <- LatticeKrig::predict.LKrig(lk0, xnew = cbind(xo, yo))
    # se <- LatticeKrig::predictSE.LKrig(lk0, xnew = cbind(xo, yo))
}