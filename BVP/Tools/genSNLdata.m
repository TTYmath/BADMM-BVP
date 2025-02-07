function [G,C,c,Sen,Anc,gnods] = genSNLdata(sig,R,N,na,t)
rng('default');
Anc = rand(na,2);
Sen = rand(N,2);
%% sensor to sensor
ds = sum(Sen.*Sen,2);
dis2 = ds+ds'-2*(Sen*Sen');
dis2 = dis2.*(dis2>0);
dis = sqrt(dis2);
disn = dis+sig*randn(N,N);
A = rand(N,N);
Po = exp(-dis2/(2*R^2));
A = A<Po;
[I,J] = find(triu(A,1)); %% random observation
G = triu(A,1)+triu(A,1)';
m = length(I);
C = cell(m,1);
c = cell(N,1);
[x,y] = ndgrid(0:t,0:t);
x = x/t; y = y/t;
x = x(:); y = y(:);
gnods = [x,y]; % all grid nodes
r = length(x);
d = sum(gnods.*gnods,2);
dismat2 = d+d'-2*(gnods*gnods');
dismat2 = dismat2.*(dismat2>0);
dismat = sqrt(dismat2);
logPomat = -dismat2/(2*R^2);
for k = 1:m
    i = I(k); j = J(k);
    logPhik = logPomat-(disn(i,j)-dismat).^2/(2*sig^2)-log(sqrt(2*pi*sig^2));
    logPhik = logPhik-logsumexp(logsumexp(logPhik,1),2);
    C{k} = -logPhik;
end
%% sensor to anchor
da = sum(Anc.*Anc,2);
dissa2 = ds+da'-2*Sen*Anc';
dissa2 = dissa2.*(dissa2>0);
dissa = sqrt(dissa2);
dissan = dissa+sig*randn(N,na);
A = rand(N,na);
Po = exp(-dissa2/(2*R^2));
A = A<Po;
logphi = cell(N,1);
dsamat2 = d+da'-2*gnods*Anc';
dsamat2 = dsamat2.*(dsamat2>0);
dsamat = sqrt(dsamat2);
logPsaomat = -dsamat2/(2*R^2);
for i = 1:N
    logphi{i} = zeros(r,1);
    for j = 1:na
        if A(i,j) == 1
            logphi{i} = logphi{i}+logPsaomat(:,j)-(dissan(i,j)-dsamat(:,j)).^2/(2*sig^2)-log(sqrt(2*pi*sig^2));
        end
    end
    logphi{i} = logphi{i}-logsumexp(logphi{i},1);
    c{i} = -logphi{i};
end













