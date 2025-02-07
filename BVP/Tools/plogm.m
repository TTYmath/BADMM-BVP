%% perturbed logm 
function logX = plogm(X)
logX = logm(X+1e-12*eye(size(X)));