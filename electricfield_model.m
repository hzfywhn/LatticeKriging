superdarn = readmatrix('E_superdarn.txt');
weimer = readmatrix('E_weimer.txt');

outline = alphaShape(superdarn(1, :)', superdarn(2, :)');
valid = ~inShape(outline, weimer(1, :), weimer(2, :));
weimer = weimer(:, valid);

len_superdarn = size(superdarn, 2);
len_weimer = size(weimer, 2);

x = [superdarn(1, :) weimer(1, :)]';
y = [superdarn(2, :) weimer(2, :)]';
vx = [superdarn(3, :) weimer(3, :)]';
vy = [superdarn(4, :) weimer(4, :)]';
vxe = [ones(len_superdarn, 1); 4*ones(len_weimer, 1)];
vye = [ones(len_superdarn, 1); 4*ones(len_weimer, 1)];
obs.loc = [x y];
obs.val = [vx vy];
obs.err = [vxe vye];

delta = pi/180;
x0 = -pi/3: delta: pi/3;
y0 = -pi/3: delta: pi/3;
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

normalization = false;
rho = 1;
derivative = true;
[y, W, Z, phi, Q] = constants(obs, basis, normalization, rho, derivative);
lambda = optimize(y, W, Z, phi, Q, exp(-10), exp(-6));
[d, c, rhoMLE, likelihood, M] = kriging(lambda, y, W, Z, phi, Q);
[m, sd] = prediction(loc, basis, normalization, rho, lambda, Z, phi, Q, M, d, c, rhoMLE);

function lambda = optimize(y, W, Z, phi, Q, xmin, xmax)
    function likelihood = LK(l)
        [~, ~, ~, likelihood] = kriging(exp(l), y, W, Z, phi, Q);
        likelihood = -likelihood;
    end
    lambda = exp(fminbnd(@LK, log(xmin), log(xmax), optimset('FunValCheck', 'on', 'TolX', eps)));
end