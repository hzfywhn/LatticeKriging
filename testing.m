x1 = -4;
x2 = 4;
y1 = -2;
y2 = 2;

delta = 0.1;
x0 = x1: delta: x2;
y0 = y1: delta: y2;
[y, x] = meshgrid(y0, x0);
z = 1 ./ (1 + (x-1).^2 + y.^2) - 1 ./ (1 + (x+1).^2 + y.^2);
vx = -2*(x-1) ./ (1 + (x-1).^2 + y.^2).^2 + 2*(x+1) ./ (1 + (x+1).^2 + y.^2).^2;
vy = -2*y ./ (1 + (x-1).^2 + y.^2).^2 + 2*y ./ (1 + (x+1).^2 + y.^2).^2;
v = sqrt(vx.^2 + vy.^2);
cosx = vx ./ v;
cosy = vy ./ v;
obs = struct('loc', [x(:) y(:)], 'azim', [cosx(:) cosy(:)], 'val', v(:), 'err', ones(numel(v), 1));

delta = 0.2;
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
derivative = true;
[y, W, Z, Q, phi] = constants(obs, basis, normalization, rho, derivative);
lambda = optimize(y, W, Z, Q, phi, exp(-9), exp(5), 5e-3);
[d, c, rhoMLE, likelihood, M] = kriging(lambda, y, W, Z, Q, phi);

delta = 0.1;
x0 = x1: delta: x2;
y0 = y1: delta: y2;
[y, x] = meshgrid(y0, x0);
z = 1 ./ (1 + (x-1).^2 + y.^2) - 1 ./ (1 + (x+1).^2 + y.^2);
vx = -2*(x-1) ./ (1 + (x-1).^2 + y.^2).^2 + 2*(x+1) ./ (1 + (x+1).^2 + y.^2).^2;
vy = -2*y ./ (1 + (x-1).^2 + y.^2).^2 + 2*y ./ (1 + (x+1).^2 + y.^2).^2;
[m, sd] = prediction([x(:) y(:)], basis, normalization, rho, lambda, Z, Q, phi, M, d, c, rhoMLE);

subplot('Position', [0.05 0.05 0.4 0.9])
h = pcolor(x, y, z);
h.EdgeColor = 'none';
hold on
quiver(x, y, -vx, -vy, 'k', 'LineWidth', 1)
hold off
title('Electric Field Input')
axis([-4 4 -2 2])
caxis([-0.8 0.8])
colormap('jet')
subplot('Position', [0.5 0.05 0.4 0.9])
h = pcolor(x, y, reshape(m, [length(x0) length(y0)]));
h.EdgeColor = 'none';
title('Electric Potential Output')
axis([-4 4 -2 2])
caxis([-0.8 0.8])
colormap('jet')
colorbar('Position', [0.92 0.05 0.03 0.9])

function lambda = optimize(y, W, Z, Q, phi, xmin, xmax, tol)
    function likelihood = LK(l)
        [~, ~, ~, likelihood] = kriging(exp(l), y, W, Z, Q, phi);
        likelihood = -likelihood;
    end
    lambda = exp(fminbnd(@LK, log(xmin), log(xmax), optimset('FunValCheck', 'on', 'TolX', tol)));
end