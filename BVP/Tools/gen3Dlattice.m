%% generate n by n 2D grid graph
function A = gen3Dlattice(n)
if n^3>1000000
    error('\n graph is too large');
end
n1 = n^3; 
[x,y,z] = ndgrid(1:n,1:n,1:n);
x = x(:); y = y(:); z = z(:);

B_l = [x,y,z,x-1,y,z];
B_r = [x,y,z,x+1,y,z];
B_u = [x,y,z,x,y-1,z];
B_d = [x,y,z,x,y+1,z];
B_f = [x,y,z,x,y,z-1];
B_b = [x,y,z,x,y,z+1];

idl = (x-1)>0;
idr = (x+1)<=n;
idu = (y-1)>0;
idd = (y+1)<=n;
idf = (z-1)>0;
idb = (z+1)<=n;

B = [B_l(idl,:);B_r(idr,:);B_u(idu,:);B_d(idd,:);B_f(idf,:);B_b(idb,:)];

I = B(:,1)+(B(:,2)-1)*n+(B(:,3)-1)*n^2;
J = B(:,4)+(B(:,5)-1)*n+(B(:,6)-1)*n^2;
b = ones(length(I),1);
A = sparse(I,J,b,n1,n1);
A = (A+A')>0; 

