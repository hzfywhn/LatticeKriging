function [ia, ja, a] = SAR(basis)
    [m, m0] = size(basis.connect);

    ia = zeros(1, m*m0);
    ja = zeros(1, m*m0);
    a = zeros(1, m*m0);

    k = 1;
    for j = 1: m
        ia(k) = j;
        ja(k) = j;
        a(k) = basis.centerweight;
        k = k + 1;

        for j0 = 1: m0
            js = basis.connect(j, j0);
            if ~isnan(js)
                ia(k) = j;
                ja(k) = js;
                a(k) = -1;
                k = k + 1;
            end
        end
    end

    ia = ia(1: k-1);
    ja = ja(1: k-1);
    a = a(1: k-1);
end