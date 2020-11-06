function [ia, ja, a] = regression(loc, basis, derivative)
    [n, ndim] = size(loc);
    m = size(basis.loc, 1);

    nr = n;
    if derivative
        nr = ndim * n;
    end

    ia = zeros(1, nr*m);
    ja = zeros(1, nr*m);
    a = zeros(1, nr*m);

    k = 1;
    for i = 1: n
        for j = 1: m
            displace = loc(i, :) - basis.loc(j, :);
            d = norm(displace) / basis.delta;
            if d <= 1
                if derivative
                    r = -(1 - d)^5 * (5*d + 1) * 56/3 / basis.delta^2;
                    for idim = 1: ndim
                        ia(k) = i + (idim-1)*n;
                        ja(k) = j;
                        a(k) = r * displace(idim);
                        k = k + 1;
                    end
                else
                    ia(k) = i;
                    ja(k) = j;
                    a(k) = (1 - d)^6 * (35*d^2 + 18*d + 3) / 3;
                    k = k + 1;
                end
            end
        end
    end

    ia = ia(1: k-1);
    ja = ja(1: k-1);
    a = a(1: k-1);
end