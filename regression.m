function [ia, ja, a] = regression(obs, basis, derivative)
    n = size(obs.loc, 1);
    m = size(basis.loc, 1);

    ia = zeros(1, n*m);
    ja = zeros(1, n*m);
    a = zeros(1, n*m);

    k = 1;
    for i = 1: n
        for j = 1: m
            displace = obs.loc(i, :) - basis.loc(j, :);
            d = norm(displace) / basis.delta;
            if d <= 1
                ia(k) = i;
                ja(k) = j;
                if derivative
                    v = -(1 - d)^5 * (5*d + 1) * 56/3 / basis.delta^2 * displace;
                    a(k) = sum(v .* obs.azim(i, :));
                else
                    a(k) = (1 - d)^6 * (35*d^2 + 18*d + 3) / 3;
                end
                k = k + 1;
            end
        end
    end

    ia = ia(1: k-1);
    ja = ja(1: k-1);
    a = a(1: k-1);
end