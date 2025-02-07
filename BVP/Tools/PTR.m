%% compute partial trace
function B = PTR(A,T) 
rr = size(A,1);
r = round(sqrt(rr));
a = A(T);
B = reshape(sum(reshape(a,r,r^2),1)',r,r);
B = (B+B')/2; 