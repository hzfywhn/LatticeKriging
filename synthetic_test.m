% a synthetic test demonstrating the procedure of Lattice Kriging

% modeling domain
x1 = -4;
x2 = 4;
y1 = -2;
y2 = 2;

% construct the location matrix and connectivity matrix of basis functions
% for simplicity, only one level of basis functions are used in this test
% all basis functions are equally spaced with connectivity matrix connecting the adjacent 4 points (left, right, top, bottom)
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

% due to the simplified geometry, grids connected to the current grid are directly obtained
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

% center weight is typically 2^d+0.01 where d is the dimension of the model, for 2D problems, d=2
% enlarge the width of basis functions to allow overlap, currently 2.5 times the spacing as suggested by Nychka et al. (2015)
basis = {struct('loc', loc, 'connect', con, 'centerweight', 4.01, 'delta', delta*2.5, 'alpha', 1)};

% normalization of the regresion matrix is turned off (no significant impact is found)
normalization = false;

% initial guess of the marginal variance can be chosen arbitrary
rho = 1;

% observation location and direction cosine are defined using random numbers
% therefore the outputs of two runs will be different, but the overall pattern will be similar
n = 1000;
x = x1 + (x2-x1)*rand(n, 1);
y = y1 + (y2-y1)*rand(n, 1);
t = 2*pi*rand(n, 1);

% the potential is designed as a two cell pattern of equal magnitude with opposite sign
vx = 2*((x-1)./(1+(x-1).^2+y.^2).^2 - (x+1)./(1+(x+1).^2+y.^2).^2);
vy = 2*(y./(1+(x-1).^2+y.^2).^2 - y./(1+(x+1).^2+y.^2).^2);

% actual observation is the line-of-sight projection of the tangent field
yfit = vx.*cos(t) + vy.*sin(t);

% error covariance matrix is set as identity
W = speye(n);

% the prior is chosen as a linear function of x, vector modeling takes the line-of-sight projection
Z = -cos(t);

% construct the precision and regression matrix
[Q, phi] = combineMR(struct('loc', [x y], 'dircos', [cos(t) sin(t)]), basis, normalization, rho, true);

% find the best estimate of lambda
[l, ~, ~, ~] = findblup(yfit, W, Z, Q, phi, -4, 4, optimset('FunValCheck', 'on'));
lambda = exp(l);

% calculate all paramters based on the best estimate of lambda
[d, c, rhoMLE, likelihood, M] = kriging(lambda, yfit, W, Z, Q, phi);

% prediction is carried out at a different resolution
delta = 0.1;
x0 = x1: delta: x2;
y0 = y1: delta: y2;
nx = length(x0);
ny = length(y0);
[yo, xo] = meshgrid(y0, x0);

% the prior of the prediction is still a linear function of x, but without line-of-sight projection
Z1 = xo(:);

% construct the regression matrix based on the new location
[~, phi1] = combineMR(struct('loc', [xo(:) yo(:)]), basis, normalization, rho, false);

% predict the mean and variance at new locations
m = predictMean(Z1, phi1, d, c);
sd = predictSD(Z1, phi1, lambda, W, Z, Q, phi, M, rhoMLE);

% save the whole process for plotting
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