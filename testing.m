x1 = -4;
x2 = 4;
y1 = -2;
y2 = 2;

delta = 0.5;
x0 = x1: delta: x2;
y0 = y1: delta: y2;
nx = length(x0);
ny = length(y0);
loc = zeros(nx*ny, 2);
con = nan(nx*ny, 4);
for j = 1: ny
    for i = 1: nx
        idx = (j-1)*nx + i;
        loc(idx, 2) = y0(j);
        loc(idx, 1) = x0(i);
        if j >= 2
            con(idx, 1) = (j-2)*nx + i;
        end
        if i >= 2
            con(idx, 2) = (j-1)*nx + i-1;
        end
        if j <= ny-1
            con(idx, 3) = j*nx + i;
        end
        if i <= nx-1
            con(idx, 4) = (j-1)*nx + i+1;
        end
    end
end
basis = {struct('loc', loc, 'connect', con, 'centerweight', 4.01, 'delta', delta*2.5, 'alpha', 1)};

normalization = false;
rho = 1;

delta = 0.1;
[y, x] = meshgrid(y1: delta: y2, x1: delta: x2);
z = 1 ./ (1 + (x-1).^2 + y.^2) - 1 ./ (1 + (x+1).^2 + y.^2);
vx = 2*(x-1) ./ (1 + (x-1).^2 + y.^2).^2 - 2*(x+1) ./ (1 + (x+1).^2 + y.^2).^2;
vy = 2*y ./ (1 + (x-1).^2 + y.^2).^2 - 2*y ./ (1 + (x+1).^2 + y.^2).^2;
n = numel(z);
subset = randsample(n, floor(n/10));
n = length(subset);
x = x(subset);
y = y(subset);
z = z(subset);
vx = vx(subset);
vy = vy(subset);
cosx = ones(n, 1);
cosy = sqrt(1 - cosx.^2);
v = vx.*cosx + vy.*cosy;

subplot(211)
scatter(x, y, [], z, 'filled')
hold on
quiver(x, y, vx.*cosx, vy.*cosy)
hold off
axis([x1 x2 y1 y2])
colorbar

yfit = v;
W = sparse(1: n, 1: n, ones(n, 1));
Z = ones(n, 1);
[Q, phi] = combineMR(struct('loc', [x y], 'azim', [cosx cosy]), basis, normalization, rho, true);
lambda = optimize(yfit, W, Z, Q, phi, exp(-9), exp(5), 5e-3);
[d, c, rhoMLE, likelihood, M] = kriging(lambda, yfit, W, Z, Q, phi);

delta = 0.1;
x0 = x1: delta: x2;
y0 = y1: delta: y2;
nx = length(x0);
ny = length(y0);
[y, x] = meshgrid(y0, x0);
[~, phi1] = combineMR(struct('loc', [x(:) y(:)]), basis, normalization, rho, false);
[m, sd] = prediction(ones(nx*ny, 1), phi1, lambda, Z, Q, phi, M, d, c, rhoMLE);

subplot(212)
h = pcolor(x, y, reshape(m, nx, ny));
h.EdgeColor = 'none';
axis([x1 x2 y1 y2])
colorbar

function lambda = optimize(y, W, Z, Q, phi, xmin, xmax, tol)
    function likelihood = LK(l)
        [~, ~, ~, likelihood] = kriging(exp(l), y, W, Z, Q, phi);
        likelihood = -likelihood;
    end
    lambda = exp(fminbnd(@LK, log(xmin), log(xmax), optimset('FunValCheck', 'on', 'TolX', tol)));
end