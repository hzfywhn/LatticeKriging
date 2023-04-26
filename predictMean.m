function m = predictMean(Z1, phi1, d, c)
% calculate the conditional mean at specified locations given existing observations
% only support predicting scalar fields for now

    m = Z1*d + phi1*c;
end