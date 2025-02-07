%% GSBP.m: Gauss-Seidel Belief propagation algorithm for Bethe variational problem
%% This implementation is based on the Belief propagation method
%% proposed by Pearl, Judea in the paper:
%% Pearl, J. (2014). Probabilistic reasoning in intelligent systems:
%% networks of plausible inference. Elsevier.
%% Please cite the above paper if you use this implementation in your work.
function runhist = GSBP(G,C,c,T,par) 
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
%% initialization 
tG = triu(G);
[I,J,~] = find(tG); 
deg = sum(G,2);
m = length(I); 
n = size(G,1); 
r = length(c{1}); 
NI = cell(1,n); 
NJ = cell(1,n); 
for k = 1:m
    i = I(k);
    j = J(k);
    NI{i} = [NI{i},k];
    NJ{j} = [NJ{j},k];
end
logPhi = cell(m,1);
logphi = cell(n,1);
for k = 1:m
    logPhi{k} = -C{k}/T;
end
for k = 1:n
    logphi{k} = -c{k}/T;
end
logMlam = cell(m,1); 
logMmu = cell(m,1);
q  = cell(n,1); 
logq = cell(n,1);
Q = cell(m,1); 
logQ = cell(m,1);
for k = 1:n
    q{k} = ones(r,1)/r;
    logq{k} = log(q{k});
end
for k = 1:m
    Q{k} = ones(r,r)/r^2;
    logQ{k} = log(Q{k});
end
for i = 1:m 
    logMlam{i} = log(ones(r,1)/r); 
    logMmu{i} = log(ones(r,1)/r); 
end 
%% main loop 
if verbose
    fprintf('\n ***** Guass Seidel Belief Propagation*********');
    fprintf('\n ');
end 
for iter = 1:maxiter 
    %% update logMlam
    for k = 1:m
        i = I(k);
        j = J(k);
        l = logphi{j};
        for h = NI{j}
            l = l+logMlam{h};
        end
        for h = NJ{j}
            if I(h)~=i
                l = l+logMmu{h};
            end
        end
        l = logsumexp(logPhi{k}+l',2);
        logMlam{k} = l-logsumexp(l,1);
    end
    %% update logMmu
    for k = 1:m
        i = I(k);
        j = J(k);
        l = logphi{i};
        for h = NI{i}
            if J(h)~=j
                l = l+logMlam{h};
            end
        end
        for h = NJ{i}
            l = l+logMmu{h};
        end
        l = logsumexp(logPhi{k}'+l',2);
        logMmu{k} = l-logsumexp(l,1);
    end
    %% compute Q and q
    for k = 1:n
        logq{k} = logphi{k};
        for h = NI{k}
            logq{k} = logq{k}+logMlam{h};
        end
        for h = NJ{k}
            logq{k} = logq{k}+logMmu{h};
        end
        logq{k} = logq{k}-logsumexp(logq{k},1);
        q{k} = exp(logq{k});
    end
    for k = 1:m
        i = I(k);
        j = J(k);
        l1 = logphi{i};
        l2 = logphi{j};
        for h = NI{i}
            if J(h)~=j
                l1 = l1+logMlam{h};
            end
        end
        for h = NJ{i}
            l1 = l1+logMmu{h};
        end
        for h = NI{j}
            l2 = l2+logMlam{h};
        end
        for h = NJ{j}
            if I(h)~=i
                l2 = l2+logMmu{h};
            end
        end
        M = (l1+logPhi{k})+l2';
        logQ{k} = M-logsumexp(logsumexp(M,1),2);
        Q{k} = exp(logQ{k});
    end
    %% compute residue
    pfeas = 0;
    for k = 1:m
        i = I(k);
        j = J(k);
        pfeas = pfeas+Prod(logq{i}-logsumexp(logQ{k},2),q{i})+Prod(logq{j}-logsumexp(logQ{k},1)',q{j});
    end
    %% compute function value
    fval = 0;
    for k = 1:m
        fval = fval+Prod(C{k},Q{k})+T*Prod(logQ{k},Q{k});
    end
    for k = 1:n 
        fval = fval+Prod(c{k},q{k})-T*(deg(k)-1)*Prod(logq{k},q{k});
    end
    ttime = etime(clock,tstart);
    if verbose
        fprintf('\n iter = %2d, fval = %6.7e, Pfeas = %3.2e, ttime = %3.2e ',iter,fval,pfeas,ttime);
    end
    if pfeas < tol
        break; 
    end
    if ttime > maxtime
        if (verbose)
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

























