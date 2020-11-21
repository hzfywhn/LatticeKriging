superdarn = readmatrix('superdarn.txt');
empirical = readmatrix('empirical.txt');

outline = alphaShape(superdarn(1, :)', superdarn(2, :)');
valid = ~inShape(outline, empirical(1, :), empirical(2, :));
empirical = empirical(:, valid);

len_superdarn = size(superdarn, 2);
len_empirical = size(empirical, 2);

x = [superdarn(1, :) empirical(1, :)]';
y = [superdarn(2, :) empirical(2, :)]';
cosx = [superdarn(3, :) empirical(3, :)]';
cosy = [superdarn(4, :) empirical(4, :)]';
v = [superdarn(5, :) empirical(5, :)]';
ve = [ones(len_superdarn, 1); ones(len_empirical, 1)];
obs.loc = [x y];
obs.azim = [cosx cosy];
obs.val = v;
obs.err = ve;

delta = pi/60;
x0 = -pi/3: delta: pi/3;
y0 = -pi/3: delta: pi/3;
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
basis = cell(1);
basis{1}.loc = loc;
basis{1}.connect = con;
basis{1}.centerweight = 4.01;
basis{1}.delta = delta * 2.5;
basis{1}.alpha = 1;

normalization = true;
rho = 1;
derivative = true;
[y, W, Z, Q, phi] = constants(obs, basis, normalization, rho, derivative);
lambda = optimize(y, W, Z, Q, phi, exp(-9), exp(5), 5e-3);
[d, c, rhoMLE, likelihood, M] = kriging(lambda, y, W, Z, Q, phi);

valid = x.^2 + y.^2 <= (pi*2/9)^2;
scatter(x(valid), y(valid), [], -m(valid(:)), 'filled')
hold on
quiver(superdarn(1, :), superdarn(2, :), superdarn(5, :).*superdarn(3, :), superdarn(5, :).*superdarn(4, :), 2, 'k')
quiver(empirical(1, :), empirical(2, :), empirical(5, :).*empirical(3, :), empirical(5, :).*empirical(4, :), 2, 'm')
hold off
axis([-pi*2/9 pi*2/9 -pi*2/9 pi*2/9])
cmax = max(abs(m));
caxis([-cmax cmax])
colormap('jet')
colorbar

delta = pi/180;
x0 = -pi*2/9: delta: pi*2/9;
y0 = -pi*2/9: delta: pi*2/9;
[y, x] = meshgrid(y0, x0);
[m, sd] = prediction([x(:) y(:)], basis, normalization, rho, lambda, Z, Q, phi, M, d, c, rhoMLE);

function lambda = optimize(y, W, Z, Q, phi, xmin, xmax, tol)
    function likelihood = LK(l)
        [~, ~, ~, likelihood] = kriging(exp(l), y, W, Z, Q, phi);
        likelihood = -likelihood;
    end
    lambda = exp(fminbnd(@LK, log(xmin), log(xmax), optimset('FunValCheck', 'on', 'TolX', tol)));
end