% generate a random graph with n vertices and m edges
% return the MD_ordering and right_neighbourhood
function [A,Ne] = random_graph_m(n,m)
if m > n*(n-1)/2
    error('\n too many edges, cannot generate!');
end
A = spalloc(n,n,m);
idx = randperm(n*(n-1)/2,m); 
idxi = ceil((3+sqrt(8*idx+1))/2)-1;
idxj = idx - (idxi-1).*(idxi-2)/2;
idxG = (idxi-1)*n+idxj;
A(idxG) = 1;
A = A+A';
[A,Ne,~,~] = MD_ordering(A);
end

