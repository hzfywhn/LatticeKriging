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
x0 = x1: delta: x2;
y0 = y1: delta: y2;
[y, x] = meshgrid(y0, x0);
n = numel(x);
sub = randsample(n, floor(n/10));
x = x(sub);
y = y(sub);
z = 1 ./ (1 + (x-1).^2 + y.^2) - 1 ./ (1 + (x+1).^2 + y.^2);
vx = -2*(x-1) ./ (1 + (x-1).^2 + y.^2).^2 + 2*(x+1) ./ (1 + (x+1).^2 + y.^2).^2;
vy = -2*y ./ (1 + (x-1).^2 + y.^2).^2 + 2*y ./ (1 + (x+1).^2 + y.^2).^2;
cosx = ones(size(vx));
cosy = zeros(size(vy));
v = vx.*cosx + vy.*cosy;
obs = struct('loc', [x y], 'azim', [cosx cosy], 'val', v, 'err', ones(numel(v), 1));

subplot(211)
scatter(x, y, [], z, 'filled')
hold on
quiver(x, y, vx.*cosx, vy.*cosy)
hold off
colorbar

y = obs.val;
ind = 1: size(obs.val, 1);
W = sparse(ind, ind, obs.err);
Z = ones(size(v));
[Q, phi] = combineMR(obs, basis, normalization, rho, true);
lambda = optimize(y, W, Z, Q, phi, exp(-9), exp(5), 5e-3);
[d, c, rhoMLE, likelihood, M] = kriging(lambda, y, W, Z, Q, phi);

delta = 0.1;
x0 = x1: delta: x2;
y0 = y1: delta: y2;
[y, x] = meshgrid(y0, x0);
z = 1 ./ (1 + (x-1).^2 + y.^2) - 1 ./ (1 + (x+1).^2 + y.^2);
vx = -2*(x-1) ./ (1 + (x-1).^2 + y.^2).^2 + 2*(x+1) ./ (1 + (x+1).^2 + y.^2).^2;
vy = -2*y ./ (1 + (x-1).^2 + y.^2).^2 + 2*y ./ (1 + (x+1).^2 + y.^2).^2;
[~, phi1] = combineMR(struct('loc', [x(:) y(:)]), basis, normalization, rho, false);
[m, sd] = prediction(ones(numel(z), 1), phi1, lambda, Z, Q, phi, M, d, c, rhoMLE);

subplot(212)
h = pcolor(x, y, reshape(m, [length(x0) length(y0)]));
h.EdgeColor = 'none';
colorbar

function lambda = optimize(y, W, Z, Q, phi, xmin, xmax, tol)
    function likelihood = LK(l)
        [~, ~, ~, likelihood] = kriging(exp(l), y, W, Z, Q, phi);
        likelihood = -likelihood;
    end
    lambda = exp(fminbnd(@LK, log(xmin), log(xmax), optimset('FunValCheck', 'on', 'TolX', tol)));
end