%% generate n by n 2D grid graph
function A = gen2Dgrid(n)
if n^2>1000000
    error('\n graph is too large');
end
n1 = n^2;
[x,y] = ndgrid(1:n,1:n);
x = x(:); y = y(:);
B_l = [x,y,x-1,y];
B_r = [x,y,x+1,y];
B_u = [x,y,x,y-1];
B_d = [x,y,x,y+1];
idl = (x-1)>0;
idr = (x+1)<=n;
idu = (y-1)>0;
idd = (y+1)<=n;
B = [B_l(idl,:);B_r(idr,:);B_u(idu,:);B_d(idd,:)];
I = B(:,1)+(B(:,2)-1)*n;
J = B(:,3)+(B(:,4)-1)*n;
b = ones(length(I),1);
A = sparse(I,J,b,n1,n1);
A = (A+A')>0;

