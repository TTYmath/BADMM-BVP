%% QBADMM.m: Quantum Bregman ADMM for quantum Bethe variational problem
%% Copyright (C) 2024 
%% Yuehaw Khoo, University of Chicago
%% Tianyun Tang and Kim-Chuan Toh, Natinoal University of Singapore
%% This program is free software: you can redistribute it and/or modify
%% it under the terms of the GNU General Public License as published by
%% the Free Software Foundation, version 3
%%
%% This program is distributed in the hope that it will be useful,
%% but WITHOUT ANY WARRANTY; without even the implied warranty of
%% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
%% GNU General Public License for more details.
%%
%% You should have received a copy of the GNU General Public License
%% along with this program. If not, see <https://www.gnu.org/licenses/>.
function runhist = QT_BADMM(G,C,c,par) 
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
[I,J,deg,m,n,r,TI,TJ] = QT_Preprocessing(G,c); 
%% initialization 
lam = cell(m,1); 
mu = cell(m,1); 
q = cell(n,1); 
Q = cell(m,1); 
logq = cell(n,1); 
logQ = cell(m,1); 
logqp = cell(n,1); 
logQp = cell(m,1); 
for k = 1:m
    lam{k} = zeros(r,r);
    mu{k} = zeros(r,r);
    Q{k} = eye(r^2)/r^2;
    logQ{k} = logm(Q{k});
end
for k = 1:n
    q{k} = eye(r)/r;
    logq{k} = logm(q{k});
end
%% main loop 
if verbose
    fprintf('\n ***** Bregman ADMM*********');
    fprintf('\n iter |fval          |pfeas   |dfeas   |beta    | time    | minq ');
end
beta = 1;  
minbeta = 1e-3;  
maxbeta = 1e3;  
inc = 5;
for iter = 1:maxiter
    %% update qk
    Sq = c; 
    minq = 1;
    for k = 1:m 
        Qk = Q{k}; 
        i = I(k); 
        j = J(k); 
        Sq{j} = Sq{j}+lam{k}-beta*logm(PTR(Qk,TI));
        Sq{i} = Sq{i}+mu{k}-beta*logm(PTR(Qk,TJ));
    end
    for k = 1:n
        Sq{k} = Sq{k}-logq{k}*(deg(k)-1);
        logq{k} = -Sq{k}/(beta*deg(k));
        [P,S] = eig(logq{k});
        d = real(diag(S));
        sd = logsumexp(d,1);
        d = d-sd;
        logq{k} = (P.*d')*P';
        d = exp(d);
        minq = min(minq,min(d));
        q{k} = (P.*d')*P';
        q{k} = (q{k}+q{k}')/2;
    end 
    %% update Qij
    for k = 1:m 
        Qk = Q{k};
        i = I(k);
        j = J(k);
        logQ{k} = -C{k}+kron(mu{k},eye(r))+kron(eye(r),lam{k})...
            +beta*(2*logQ{k}+kron(logq{i}-logm(PTR(Qk,TJ)),eye(r))+kron(eye(r),logq{j}-logm(PTR(Qk,TI))));
        logQ{k} = logQ{k}/(1+2*beta);
        [P,S] = eig(logQ{k});
        d = real(diag(S));
        sd = logsumexp(d,1);
        d = d-sd;
        logQ{k} = (P.*d')*P';
        d = exp(d);
        minq = min(minq,min(d));
        Q{k} = (P.*d')*P';
        Q{k} = (Q{k}+Q{k}')/2;
    end
    %% update lam, mu
    for k = 1:m
        Qk = Q{k};
        i = I(k);
        j = J(k);
        lam{k} = lam{k}-beta*(logm(PTR(Qk,TI))-logq{j});
        mu{k} = mu{k}-beta*(logm(PTR(Qk,TJ))-logq{i});
    end
    %% compute residue 
    if mod(iter, 10) == 1
        for k = 1:m
            logQp{k} = -C{k}+kron(mu{k},eye(r))+kron(eye(r),lam{k});
            [P,S] = eig(logQp{k});
            d = real(diag(S));
            sd = logsumexp(d,1);
            d = d-sd;
            logQp{k} = (P.*d')*P';
        end
        Sq = c;
        for k = 1:m
            i = I(k);
            j = J(k);
            Sq{j} = Sq{j}+lam{k};
            Sq{i} = Sq{i}+mu{k};
        end
        for k = 1:n
            logqp{k} = Sq{k}/(deg(k)-1);
            [P,S] = eig(logqp{k});
            d = real(diag(S));
            sd = logsumexp(d,1);
            d = d-sd; 
            logqp{k} = (P.*d')*P'; 
        end
        pfeas = 0;
        dfeas = 0;
        for k = 1:m
            Qk = Q{k};
            i = I(k);
            j = J(k);
            pfeas = pfeas+Prod(logq{i}-logm(PTR(Qk,TJ)),q{i})+Prod(logq{j}-logm(PTR(Qk,TI)),q{j});
            dfeas = dfeas+Prod(logQ{k}-logQp{k},Q{k});
        end
        for k = 1:n
            if deg(k)>1
                dfeas = dfeas+Prod(logq{k}-logqp{k},q{k});
            else
                dfeas = norm(Sq{k}-eye(r)*trace(Sq{k})/r,'fro')/(1+norm(c{k},'fro'));
            end
        end
        %% compute function value
        fval = 0;
        for k = 1:m
            fval = fval+Prod(C{k},Q{k})+Prod(logQ{k},Q{k});
        end
        for k = 1:n
            fval = fval+Prod(c{k},q{k})-(deg(k)-1)*Prod(logq{k},q{k});
        end
        fval = real(fval); pfeas = real(pfeas); dfeas = real(dfeas);
        ttime = etime(clock,tstart);
        if ttime > maxtime
            if (verbose)
                fprintf('\n reach maxtime: %3.2e',maxtime);
            end
            break;
        end
        if (verbose)
            fprintf('\n %2d  | %6.7e|%3.2e|%3.2e|%3.2e|%3.2e |%3.2e',...
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
ttime = etime(clock,tstart);
runhist.Q = Q;
runhist.q = q;
runhist.lam = lam;
runhist.mu = mu;
runhist.ttime = ttime;
runhist.iter = iter;
runhist.pfeas = pfeas;
runhist.dfeas = dfeas;
runhist.fval = fval;

