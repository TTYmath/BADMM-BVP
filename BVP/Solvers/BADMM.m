%% BADMM.m: Bregman ADMM for Bethe variational problem
%% Copyright (C) 2024 
%% Yuehaw Khoo, University of Chicago
%% Tianyun Tang and Kim-Chuan Toh, Natinoal University of Singapore
%% This program is free software: you can redistribute it and/or modify
%% it under the terms of the GNU General Public License as published by
%% the Free Software Foundation, version 3 of the License
%%
%% This program is distributed in the hope that it will be useful,
%% but WITHOUT ANY WARRANTY; without even the implied warranty of
%% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
%% GNU General Public License for more details.
%%
%% You should have received a copy of the GNU General Public License
%% along with this program. If not, see <https://www.gnu.org/licenses/>.
function runhist = BADMM(G,C,c,T,par)  
tstart = clock;  
rng('default'); 
%% parameters 
maxiter = 100;  
maxtime = 3600; 
verbose = 1; 
tol = 1e-6; 
if isfield(par,'maxiter'); maxiter = par.maxiter; end 
if isfield(par,'maxtime'); maxtime = par.maxtime; end 
if isfield(par,'verbose'); verbose = par.verbose; end 
if isfield(par,'tol'); tol = par.tol; end 
%% preprocessing 
Param = Preprocessing(G,C,c,T); 
m = Param.m; 
n = Param.n; 
r = Param.r;  
C = Param.Cmat;
c = Param.cmat; 
d = Param.d;
I = Param.I;
J = Param.J;
idx = (1:m)';
AI = sparse(I,idx,ones(m,1),n,m)';
AJ = sparse(J,idx,ones(m,1),n,m)';
%% initialization  
q = ones(r,n)/r;  
Q = ones(r,r,m)/r^2;  
logq = log(q);  
logQ = log(Q);  
lam = zeros(1,r,m);  
mu = zeros(r,1,m);  
%% main loop  
if verbose  
    fprintf('\n ***** Bregman ADMM*********');  
    fprintf('\n iter |fval          |pfeas   |dfeas   |beta     |time    | minq ');  
end 
beta = 1;  
inc = 5;
minbeta = 1e-3;  
maxbeta = 1e3;  
for iter = 1:maxiter 
    %% update q
    Sq = c; 
    Sq = Sq+squeeze(lam-beta*logsumexp(logQ,1))*AJ+squeeze(mu-beta*logsumexp(logQ,2))*AI-T*logq.*(d-1)';
    logq = -Sq./(beta*d)';
    logq = logq-logsumexp(logq,1);
    q = exp(logq); 
    minq = min(q(:)); 
    %% update Q
    logQ = (-C+mu)+lam... 
        +beta*(2*logQ+(reshape(logq(:,I),r,1,m)-logsumexp(logQ,2))+(reshape(logq(:,J),1,r,m)-logsumexp(logQ,1))); 
    logQ = logQ/(T+2*beta); 
    logQ = logQ-logsumexp(logsumexp(logQ,1),2); 
    Q = exp(logQ); 
    %% update lam, mu
    lam = lam-beta*(logsumexp(logQ,1)-reshape(logq(:,J),1,r,m)); %% nonlinear dual update, better
    mu = mu-beta*(logsumexp(logQ,2)-reshape(logq(:,I),r,1,m));
    %lam = lam-beta*(sum(Q,1)-reshape(q(:,J),1,r,m)); %% linear dual update
    %mu = mu-beta*(sum(Q,2)-reshape(q(:,I),r,1,m)); 
    %% compute residue 
    if mod(iter, 10) == 1
        logQp = (-C+mu+lam)/T; 
        logQp = logQp-logsumexp(logsumexp(logQp,1),2); 
        Sq = c+squeeze(lam)*AJ+squeeze(mu)*AI;
        logqp = Sq./(T*(d-1)');
        logqp = logqp-logsumexp(logqp,1);
        pfeas = Prod(logq(:,I)-squeeze(logsumexp(logQ,2)),q(:,I))+...
            Prod(logq(:,J)-squeeze(logsumexp(logQ,1)),q(:,J));
        dfeas = Prod(logQ-logQp,Q);
        for k = 1:n
            if d(k)>1
                dfeas = dfeas+Prod(logq(:,k)-logqp(:,k),q(:,k));
            else
                dfeas = norm(Sq(:,k)-ones(r)*sum(Sq(:,k))/r,'fro')/(1+norm(c(:,k),'fro'));
            end
        end
        %% compute function value
        fval = Prod(C,Q)+T*Prod(logQ,Q)+Prod(c,q)-T*sum(logq.*q,1)*(d-1);
        ttime = etime(clock,tstart);
        if ttime > maxtime
            if (verbose)
                fprintf('\n reach maxtime: %3.2e',maxtime);
            end
            break;
        end
        if (verbose)
            fprintf('\n %2d  | %6.7e|%3.2e|%3.2e|%3.2e |%3.2e|%3.2e',...
                iter,fval,pfeas,dfeas,beta,ttime,minq);
        end 
        if max(pfeas,dfeas) < tol
            break;
        end
        %% update beta
        if pfeas < dfeas/inc
            beta = max(beta/1.2,minbeta);
            inc = inc*1.1;
        elseif pfeas > inc*dfeas 
            beta = min(beta*1.2,maxbeta);
            inc = inc*1.1;
        end
    end
end
runhist.q = q;  
runhist.Q = Q;
runhist.fval = fval; 
runhist.iter = iter; 
runhist.ttime = etime(clock,tstart);
runhist.pfeas = pfeas;
runhist.dfeas = dfeas;






















