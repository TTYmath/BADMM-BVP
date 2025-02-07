function Param = Preprocessing(G,C,c,T)
[I,J,~] = find(triu(G));
d = sum(G,2);
m = length(I);
n = size(G,1);
r = length(c{1});
Cmat = zeros(r,r,m); 
for k = 1:m 
    Cmat(:,:,k) = C{k}; 
end 
cmat = zeros(r,n); 
for k = 1:n 
    cmat(:,k) = c{k}; 
end
idx = (1:m)';
AI = sparse(I,idx,ones(m,1),n,m)';
AJ = sparse(J,idx,ones(m,1),n,m)';
nC = sqrt(Norm(Cmat)^2+Norm(cmat)^2);
dI = sum(AI,1)'; 
dJ = sum(AJ,1)'; 
%% parameters
Param.I = I;
Param.J = J;
Param.AI = AI;
Param.AJ = AJ;
Param.m = m;
Param.n = n;
Param.r = r;
Param.d = d;
Param.T = T;
Param.C = C;
Param.c = c;
Param.Cmat = Cmat;
Param.cmat = cmat;
Param.nC = nC;
Param.maxsinkiter = 10000;
Param.sub = 0; 
Param.dI = dI;
Param.dJ = dJ;
Param.ep = 1e-300; 
Param.ptol = 1e-10;
Param.pert = 1e-8;



