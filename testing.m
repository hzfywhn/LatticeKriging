x1 = -2;
x2 = 2;
y1 = -1;
y2 = 1;

delta = 0.1;
x0 = x1: delta: x2;
y0 = y1: delta: y2;
[y, x] = meshgrid(y0, x0);
z = 1 ./ (1 + (x-1).^2 + y.^2) - 1 ./ (1 + (x+1).^2 + y.^2);
vx = -2*(x-1) ./ (1 + (x-1).^2 + y.^2).^2 + 2*(x+1) ./ (1 + (x+1).^2 + y.^2).^2;
vy = -2*y ./ (1 + (x-1).^2 + y.^2).^2 + 2*y ./ (1 + (x+1).^2 + y.^2).^2;
obs.loc = [x(:) y(:)];
obs.val = z(:);
obs.err = ones(numel(z), 1);

basis = cell(2);

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
basis{1}.loc = loc;
basis{1}.connect = con;
basis{1}.centerweight = 4.01;
basis{1}.delta = delta * 2.5;
basis{1}.alpha = 0.8;

delta = 0.1;
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
basis{2}.loc = loc;
basis{2}.connect = con;
basis{2}.centerweight = 4.01;
basis{2}.delta = delta * 2.5;
basis{2}.alpha = 0.2;

normalization = true;
rho = 1;
derivative = false;
[y, W, Z, Q, phi] = constants(obs, basis, normalization, rho, derivative);
lambda = optimize(y, W, Z, Q, phi, exp(-9), exp(5), 5e-3);
[d, c, rhoMLE, likelihood, M] = kriging(lambda, y, W, Z, Q, phi);

delta = 0.1;
x0 = x1: delta: x2;
y0 = y1: delta: y2;
[y, x] = meshgrid(y0, x0);
z = 1 ./ (1 + (x-1).^2 + y.^2) - 1 ./ (1 + (x+1).^2 + y.^2);
[m, sd] = prediction([x(:) y(:)], basis, normalization, rho, lambda, Z, Q, phi, M, d, c, rhoMLE);

function lambda = optimize(y, W, Z, Q, phi, xmin, xmax, tol)
    function likelihood = LK(l)
        [~, ~, ~, likelihood] = kriging(exp(l), y, W, Z, Q, phi);
        likelihood = -likelihood;
    end
    lambda = exp(fminbnd(@LK, log(xmin), log(xmax), optimset('FunValCheck', 'on', 'TolX', tol)));
end