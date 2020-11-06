x1 = -2;
x2 = 2;
y1 = -1;
y2 = 1;

delta = 0.2;
x0 = x1: delta: x2;
y0 = y1: delta: y2;
[y, x] = meshgrid(y0, x0);
z = 1 ./ (1 + (x-1).^2 + y.^2) - 1 ./ (1 + (x+1).^2 + y.^2);
vx = -2*(x-1) ./ (1 + (x-1).^2 + y.^2).^2 + 2*(x+1) ./ (1 + (x+1).^2 + y.^2).^2;
vy = -2*y ./ (1 + (x-1).^2 + y.^2).^2 + 2*y ./ (1 + (x+1).^2 + y.^2).^2;
loc = [x(:) y(:)];
obs.loc = loc;
obs.val = z(:);
obs.err = ones(numel(z), 1);

delta = 0.5;
x0 = x1: delta: x2;
y0 = y1: delta: y2;
nx = length(x0);
ny = length(y0);
[y, x] = meshgrid(y0, x0);

con = nan(nx*ny, 4);
for i = 1: nx
    for j = 1: ny
        idx = (i-1)*ny + j;
        if i >= 2
            con(idx, 1) = (i-2)*ny + j;
        end
        if j >= 2
            con(idx, 2) = (i-1)*ny + j-1;
        end
        if i <= nx-1
            con(idx, 3) = i*ny + j;
        end
        if j <= ny-1
            con(idx, 4) = (i-1)*ny + j+1;
        end
    end
end
basis = cell(1);
basis{1}.loc = [x(:) y(:)];
basis{1}.connect = con;
basis{1}.centerweight = 4.01;
basis{1}.delta = delta * 2.5;
basis{1}.alpha = 1;

normalization = true;
rho = 1;
derivative = false;
[y, W, Z, phi, Q] = constants(obs, basis, normalization, rho, derivative);
lambda = optimize(y, W, Z, phi, Q);
[d, c, rhoMLE, likelihood, M] = kriging(lambda, y, W, Z, phi, Q);
[m, sd] = prediction(loc, basis, normalization, rho, lambda, Z, phi, Q, M, d, c, rhoMLE);

function lambda = optimize(y, W, Z, phi, Q)
    function likelihood = LK(l)
        [~, ~, ~, likelihood] = kriging(exp(l), y, W, Z, phi, Q);
        likelihood = -likelihood;
    end
    lambda = exp(fminbnd(@LK, -12, -8, optimset('FunValCheck', 'on', 'TolX', eps)));
end