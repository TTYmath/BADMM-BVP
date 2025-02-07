%% CCCP.m: double-loop algorithm for Bethe variational problem
%% This implementation is based on the Concave-Convex Procedure (CCCP) 
%% proposed by A. L. Yuille in the paper:
%% Yuille, A. L. (2002). CCCP algorithms to minimize the Bethe and 
%% Kikuchi free energies: Convergent alternatives to belief propagation. 
%% Neural computation, 14(7), 1691-1722.
%% Please cite the above paper if you use this implementation in your work.
function runhist = CCCP(G,C,c,T,par) 
tstart = clock; 
rng('default'); 
%% parameters 
maxiter = 100; 
maxiniter = 5;
maxtime = 3600;
verbose = 1; 
tol = 1e-6;  
if isfield(par,'maxiter'); maxiter = par.maxiter; end 
if isfield(par,'maxtime'); maxtime = par.maxtime; end
if isfield(par,'maxiniter'); maxiniter = par.maxiniter; end
if isfield(par,'verbose'); verbose = par.verbose; end 
if isfield(par,'tol'); tol = par.tol; end 
%% initialization
tG = triu(G);
[I,J,~] = find(tG); 
deg = sum(G,2);
m = length(I); 
n = size(G,1); 
NI = cell(1,n); 
NJ = cell(1,n); 
for k = 1:m
    i = I(k);
    j = J(k);
    NI{i} = [NI{i},k];
    NJ{j} = [NJ{j},k];
end
r = length(c{1});  
q = cell(1,n);
logq = cell(1,n);
Q = cell(1,m);
logQ = cell(1,m);
logqp = cell(1,n);
qp = cell(1,n);
ch = cell(1,n);
for k = 1:n
    q{k} = ones(r,1)/r;
    logq{k} = log(q{k});
end
for k = 1:m
    Q{k} = ones(r,r)/r^2;
    logQ{k} = log(Q{k});
end
fval = 0;
for k = 1:m
    fval = fval+Prod(C{k},Q{k})+T*Prod(logQ{k},Q{k});
end
for k = 1:n
    fval = fval+Prod(c{k},q{k})-T*(deg(k)-1)*Prod(logq{k},q{k});
end
lam = cell(1,m);
mu = cell(1,m);
alp = cell(1,m);
for i = 1:m
    alp{i} = 0;
    lam{i} = zeros(r,1);
    mu{i} = zeros(r,1);
end
%% main loop 
if verbose 
    fprintf('\n ***** Convex Concave Procedure*********');
    fprintf('\n iter |fval          |reldiff    |pfeas     |dfeas     |time      |initer|'); 
end 
Initer = 0; 
for iter = 1:maxiter 
    %% linearize concave part
    for k = 1:n
        ch{k} = c{k}-T*deg(k)*(1+logq{k});
    end
    %% inner loop 
    for initer = 1:maxiniter
        %% Update alp
        for k = 1:m
            M = (-C{k}+lam{k}+mu{k}')/T-1;
            alp{k} = -T*logsumexp(logsumexp(M,1),2);
        end
        %% Update lam
        for k = 1:m
            M = (-C{k}+mu{k}'+alp{k})/T-1; 
            l1 = logsumexp(M,2);
            i = I(k);
            j = J(k);
            l2 = -ch{i};
            for h = NI{i}
                if J(h)~=j
                    l2 = l2-lam{h};
                end
            end
            for h = NJ{i}
                l2 = l2-mu{h};
            end
            l2 = l2/T-1;
            lam{k} = T*(l2-l1)/2;
        end
        %% Update mu
        for k = 1:m
            M = ((-C{k}+lam{k}+alp{k})/T-1)';
            l1 = logsumexp(M,2);
            i = I(k);
            j = J(k);
            l2 = -ch{j};
            for h = NI{j}
                l2 = l2-lam{h};
            end
            for h = NJ{j}
                if I(h)~=i
                    l2 = l2-mu{h};
                end
            end
            l2 = l2/T-1;
            mu{k} = T*(l2-l1)/2;
        end
        %% recover Q and q 
        for k = 1:n
            logq{k} = -ch{k}/T-1;
            logqp{k} = c{k};
        end
        for k = 1:m
            i = I(k);
            j = J(k);
            logQk = (-C{k}+lam{k}+mu{k}'+alp{k})/T-1;
            logQ{k} = logQk-logsumexp(logsumexp(logQk,1),2);
            Q{k} = exp(logQ{k});
            logq{i} = logq{i}-lam{k}/T;
            logq{j} = logq{j}-mu{k}/T;
            logqp{i} = logqp{i}+lam{k};
            logqp{j} = logqp{j}+mu{k};
        end
        for k = 1:n
            logq{k} = logq{k}-logsumexp(logq{k},1);
            q{k} = exp(logq{k});
            %q{k} = q{k}/sum(q{k});
            if deg(k)>1
                logqp{k} = logqp{k}/(T*(deg(k)-1));
                logqp{k} = logqp{k}-logsumexp(logqp{k},1);
                qp{k} = exp(logqp{k});
            end
        end
        %% compute residue
        pfeas = 0;
        for k = 1:m
            i = I(k);
            j = J(k);
            pfeas = pfeas+Prod(logq{i}-logsumexp(logQ{k},2),q{i})+Prod(logq{j}-logsumexp(logQ{k},1)',q{j});
        end 
        dfeas = 0;
        for k = 1:n
            if deg(k)>1
                dfeas = dfeas+Prod(logq{k}-logqp{k},q{k});
            else
                dfeas = dfeas+norm(logqp{k}-sum(logqp{k})/r,'fro')/(1+norm(c{k},'fro'));
            end
        end
    end
    Initer = Initer+initer;
    fval0 = fval;
    fval = 0;
    for k = 1:m
        fval = fval+Prod(C{k},Q{k})+T*Prod(logQ{k},Q{k});
    end
    for k = 1:n 
        fval = fval+Prod(c{k},q{k})-T*(deg(k)-1)*Prod(logq{k},q{k});
    end
    reldiff = (fval-fval0)/(1+abs(fval0));
    ttime = etime(clock,tstart);
    if verbose
        fprintf('\n %2d  | %6.7e | %3.2e | %3.2e | %3.2e | %3.2e | %2d',...
            iter,fval,reldiff,pfeas,dfeas,ttime,Initer);
    end
    %% checking stopping criterion
    if (pfeas<tol)&&(dfeas<tol)
        if verbose
            fprintf('\n convergent');
        end
        break;
    end 
    if ttime > maxtime
        if verbose
            fprintf('\n reach maxtime: %3.2e',maxtime);
        end
        break;
    end
    if Initer >= maxiter
        if verbose
            fprintf('\n reach maxiter: %2d',maxiter);
        end
        break;
    end
end
ttime = etime(clock,tstart); 
runhist.Q = Q;
runhist.q = q; 
runhist.ttime = ttime; 
runhist.iter = iter;
runhist.Initer = Initer;
runhist.pfeas = pfeas;
runhist.dfeas = dfeas;
runhist.fval = fval;







