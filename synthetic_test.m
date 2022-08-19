x1 = -4;
x2 = 4;
y1 = -2;
y2 = 2;

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
        loc(idx, 1) = x0(i);
        loc(idx, 2) = y0(j);
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

n = 1000;
x = x1 + (x2-x1)*rand(n, 1);
y = y1 + (y2-y1)*rand(n, 1);
t = 2*pi*rand(n, 1);

vx = 2*((x-1)./(1+(x-1).^2+y.^2).^2 - (x+1)./(1+(x+1).^2+y.^2).^2);
vy = 2*(y./(1+(x-1).^2+y.^2).^2 - y./(1+(x+1).^2+y.^2).^2);

yfit = vx.*cos(t) + vy.*sin(t);
W = speye(n);
Z = -cos(t);

[Q, phi] = combineMR(struct('loc', [x y], 'dircos', [cos(t) sin(t)]), basis, normalization, rho, true);

[l, ~, ~, ~] = findblup(yfit, W, Z, Q, phi, -4, 4, optimset('FunValCheck', 'on'));
lambda = exp(l);
[d, c, rhoMLE, likelihood, M] = kriging(lambda, yfit, W, Z, Q, phi);

delta = 0.1;
x0 = x1: delta: x2;
y0 = y1: delta: y2;
nx = length(x0);
ny = length(y0);
[yo, xo] = meshgrid(y0, x0);

Z1 = xo(:);
[~, phi1] = combineMR(struct('loc', [xo(:) yo(:)]), basis, normalization, rho, false);
m = predictMean(Z1, phi1, d, c);
sd = predictSD(Z1, phi1, lambda, W, Z, Q, phi, M, rhoMLE);

filename = 'synthetic_test.nc';
nccreate(filename, 'x', 'Dimensions', {'x', nx}, 'Format', 'netcdf4')
nccreate(filename, 'y', 'Dimensions', {'y', ny})
nccreate(filename, 'input_pot', 'Dimensions', {'x', nx, 'y', ny})
nccreate(filename, 'input_loc', 'Dimensions', {'n', n, 'dim', 2})
nccreate(filename, 'input_e', 'Dimensions', {'n', n, 'dim', 2})
nccreate(filename, 'background_pot', 'Dimensions', {'x', nx, 'y', ny})
nccreate(filename, 'output_mean', 'Dimensions', {'x', nx, 'y', ny})
nccreate(filename, 'output_sd', 'Dimensions', {'x', nx, 'y', ny})
ncwrite(filename, 'x', x0)
ncwrite(filename, 'y', y0)
ncwrite(filename, 'input_pot', 1./(1+(xo-1).^2+yo.^2) - 1./(1+(xo+1).^2+yo.^2))
ncwrite(filename, 'input_loc', [x y])
ncwrite(filename, 'input_e', [vx.*cos(t) vy.*sin(t)])
ncwrite(filename, 'background_pot', xo)
ncwrite(filename, 'output_mean', reshape(m, nx, ny))
ncwrite(filename, 'output_sd', reshape(sd, nx, ny))