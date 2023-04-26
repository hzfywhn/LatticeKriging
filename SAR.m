function [ia, ja, a] = SAR(basis)
% construct the spatial autoregression matrix (B) of one level
% basis: input, struct containing the connectivity matrix and the center weight (usually 2^d+0.01, d is dimension)
% ia, ja, a: output, triplet indicating the indices and values of a matrix to be used in combineMR

    [m, m0] = size(basis.connect);

    ia = zeros(1, m*m0);
    ja = zeros(1, m*m0);
    a = zeros(1, m*m0);

    k = 1;
    for j = 1: m

% the weight of the center grid is positive
        ia(k) = j;
        ja(k) = j;
        a(k) = basis.centerweight;
        k = k + 1;

% include the neighboring grids in calculation if they are adjacent (connect is not nan)
% the weights of the neighboring grids are -1
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