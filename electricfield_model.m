superdarn = readmatrix('E_superdarn.txt');
empirical = readmatrix('E_empirical.txt');

outline = alphaShape(superdarn(1, :)', superdarn(2, :)');
valid = ~inShape(outline, empirical(1, :), empirical(2, :));
empirical = empirical(:, valid);

len_superdarn = size(superdarn, 2);
len_empirical = size(empirical, 2);

x = [superdarn(1, :) empirical(1, :)]';
y = [superdarn(2, :) empirical(2, :)]';
vx = [superdarn(3, :) empirical(3, :)]';
vy = [superdarn(4, :) empirical(4, :)]';
vxe = [ones(len_superdarn, 1); ones(len_empirical, 1)];
vye = [ones(len_superdarn, 1); ones(len_empirical, 1)];
obs.loc = [x y];
obs.val = [vx vy];
obs.err = [vxe vye];

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