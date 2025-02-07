%% QT_GSBP.m: Gauss-Seidel Belief propagation algorithm for Quantum 
%% Bethe variational problem
%% This implementation is based on the Quantum belief propagation method
%% proposed by Zhan, Jian, Roberto Bondesan and Wayne Luk in the paper:
%% Zhao, J., Bondesan, R., & Luk, W. (2024). Quantum Belief Propagation.
%% Please cite the above paper if you use this implementation in your work.
function runhist = QT_GSBP(G,C,c,par) 
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
[I,J,deg,m,n,r,TI,TJ,NI,NJ] = QT_Preprocessing(G,c); 
%% initialization 
lam = cell(m,1); 
mu = cell(m,1); 
q = cell(n,1); 
Q = cell(m,1); 
logq = cell(n,1); 
logQ = cell(m,1);  
for k = 1:m 
    lam{k} = zeros(r,r);
    mu{k} = zeros(r,r); 
    Q{k} = eye(r^2)/r^2; 
    logQ{k} = plogm(Q{k}); 
end
for k = 1:n
    q{k} = eye(r)/r;
    logq{k} = plogm(q{k});
end
%% main loop 
if verbose
    fprintf('\n ***** Gauss Seidel Quantum Belief Propagation*********\n');
end
for iter = 1:maxiter 
    %% update lam
    minq = 1;
    for k = 1:m
        i = I(k);
        j = J(k);
        lam{k} = -c{j};
        for h = NJ{j}
            if I(h)~=i
                lam{k} = lam{k}+plogm(PTR(Q{h},TI))-lam{h};
            end
        end
        for h = NI{j}
            lam{k} = lam{k}+plogm(PTR(Q{h},TJ))-mu{h};
        end
    end
    %% update mu
    for k = 1:m
        i = I(k);
        j = J(k);
        mu{k} = -c{i};
        for h = NJ{i}
            mu{k} = mu{k}+plogm(PTR(Q{h},TI))-lam{h};
        end
        for h = NI{i}
            if J(h)~=j
                mu{k} = mu{k}+plogm(PTR(Q{h},TJ))-mu{h};
            end
        end
    end
    for k = 1:m
        lam{k} = (lam{k}+lam{k}')/2;
        [P,S] = eig(lam{k});
        d = diag(S);
        sd = logsumexp(d,1);
        d = d-sd;
        lam{k} = (P.*d')*P';
        mu{k} = (mu{k}+mu{k}')/2;
        [P,S] = eig(mu{k});
        d = diag(S);
        sd = logsumexp(d,1);
        d = d-sd;
        mu{k} = (P.*d')*P';
    end
    %% update Q
    for k = 1:m 
        logQ{k} = -C{k}+kron(mu{k},eye(r))+kron(eye(r),lam{k});
        Q{k} = expm(logQ{k});
        Q{k} = Q{k}/trace(Q{k});
        Q{k} = (Q{k}+Q{k}')/2;
        logQ{k} = plogm(Q{k});
    end
    %% update q 
    logq = c; 
    for k = 1:m 
        i = I(k); 
        j = J(k); 
        logq{i} = logq{i}+mu{k}; 
        logq{j} = logq{j}+lam{k}; 
    end 
    for k = 1:n 
        if deg(k)>1
            logq{k} = logq{k}/(deg(k)-1);
            q{k} = expm(logq{k});
            q{k} = q{k}/trace(q{k});
            q{k} = (q{k}+q{k}')/2;
            logq{k} = plogm(q{k});
        else
            if ~isempty(NI{k})
                j = NI{k};
                q{k} = PTR(Q{j},TJ);
                q{k} = q{k}/trace(q{k});
                q{k} = (q{k}+q{k}')/2;
                logq{k} = plogm(q{k});
            else
                i = NJ{k};
                q{k} = PTR(Q{i},TI);
                q{k} = q{k}/trace(q{k});
                q{k} = (q{k}+q{k}')/2;
                logq{k} = plogm(q{k});
            end
        end
        minq = min(minq,min(diag(q{k})));
    end   
    %% compute residue
    pfeas = 0;
    for k = 1:m
        Qk = Q{k};
        i = I(k);
        j = J(k);
        pfeas = pfeas+Prod(logq{i}-plogm(PTR(Qk,TJ)),q{i})+Prod(logq{j}-plogm(PTR(Qk,TI)),q{j});
    end
    %% compute function value
    fval = 0;
    for k = 1:m
        fval = fval+Prod(C{k},Q{k})+Prod(logQ{k},Q{k});
    end
    for k = 1:n
        fval = fval+Prod(c{k},q{k})-(deg(k)-1)*Prod(logq{k},q{k});
    end
    if isnan(fval)
        if verbose
            fprintf('\n numerical issue occurs, terminate');
        end
        break;
    end
    ttime = etime(clock,tstart);
    if mod(iter,10) == 1
        if verbose
            fprintf(' iter = %2d, fval = %6.7e, Pfeas = %3.2e, ttime = %3.2e, minq = %3.2e \n',iter,fval,pfeas,ttime,minq);
        end
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
runhist.dfeas = 0;
runhist.fval = fval;

