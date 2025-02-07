%% JBP.m: Jacobi Belief propagation algorithm for Bethe variational problem
%% This implementation is based on the Belief propagation method
%% proposed by Pearl, Judea in the paper:
%% Pearl, J. (2014). Probabilistic reasoning in intelligent systems:
%% networks of plausible inference. Elsevier.
%% Please cite the above paper if you use this implementation in your work.
function runhist = JBP(G,C,c,T,par) 
tstart = clock; 
rng('default'); 
%% parameters 
maxiter = 100; 
maxtime = 3600; 
verbose = 1; 
tol = 1e-6; 
damping = 0; 
if isfield(par,'maxiter'); maxiter = par.maxiter; end 
if isfield(par,'maxtime'); maxtime = par.maxtime; end
if isfield(par,'verbose'); verbose = par.verbose; end 
if isfield(par,'tol'); tol = par.tol; end 
if isfield(par,'damping'); damping = par.damping; end
%% initialization
[I,J,~] = find(triu(G));
deg = sum(G,2);
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
logPhi = -Cmat/T-1;
logphi = -cmat/T-1;
logMlam = zeros(r,1,m); %Mlam = exp(-lam/T)
logMmu = zeros(1,r,m); % Mmu = exp(-mu/T)
logMlam = logMlam-logsumexp(logMlam,1);
logMmu = logMmu-logsumexp(logMmu,2);
%% main loop
if verbose
    fprintf('\n ***** Jacob Belief Propagation*********');
end
for iter = 1:maxiter
    %% update d
    logd = squeeze(logMmu)*AJ+squeeze(logMlam)*AI;
    %% update M1, M2
    ll = reshape(logphi(:,I)+logd(:,I),[r,1,m])-logMlam;
    logMmup = logsumexp(logPhi+ll,1);
    ll = reshape(logphi(:,J)+logd(:,J),[1,r,m])-logMmu;
    logMlamp = logsumexp(logPhi+ll,2);
    logMmu = logMmup*(1-damping)+logMmu*damping;
    logMlam = logMlamp*(1-damping)+logMlam*damping;
    logMlam = logMlam-logsumexp(logMlam,1);
    logMmu = logMmu-logsumexp(logMmu,2);
    %% FP residue
    logq = logd+logphi;
    logq = logq-logsumexp(logq,1);
    l1 = reshape(logphi(:,I)+logd(:,I),[r,1,m])-logMlam; 
    l2 = reshape(logphi(:,J)+logd(:,J),[1,r,m])-logMmu; 
    logQ = (l1+logPhi)+l2;
    logQ = logQ-logsumexp(logsumexp(logQ,1),2);
    %% check feasibility
    Q = exp(logQ);
    q = exp(logq);
    pfeas = Prod(logq(:,I)-squeeze(logsumexp(logQ,2)),q(:,I))+...
            Prod(logq(:,J)-squeeze(logsumexp(logQ,1)),q(:,J));
    %% compute function value
    fval = Prod(cmat,q)+Prod(Cmat,Q)+T*(Prod(logQ,Q)-sum(logq.*q,1)*(deg-1));
    ttime = etime(clock,tstart); 
    if mod(iter,10) == 1
        if verbose
            fprintf('\n iter = %2d, fval = %6.7e, Pfeas = %3.2e, ttime = %3.2e ',iter,fval,pfeas,ttime);
        end
    end
    if pfeas < tol
        break;
    end
    if ttime> maxtime
        if verbose
            fprintf('\n reach maxtime: %3.2e',maxtime);
        end
        break;
    end
end
ttime = etime(clock,tstart);
runhist.Q = Q;
runhist.q = q; 
runhist.ttime = ttime;
runhist.iter = iter;
runhist.pfeas = pfeas;
runhist.fval = fval;
























