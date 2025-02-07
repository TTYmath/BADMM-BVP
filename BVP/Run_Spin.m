%% Test 2D, 3D spin glass model 
clear all 
close all 
addpath('./Tools'); 
addpath('./Solvers/'); 
rng('default'); 
%% parameters
par.maxiter = 10000;
par.tol = 1e-6;
par.verbose = 1;
par.maxtime = 3600;
Atype = 1; % 1, 2D grid; 2, 3D grid; 3, random 
r = 2; % spin model: r=2 
T = 1; % spin model: T=1 
%% solvers
useJBP = 0; %% Jacobian Belief Propagation
useGSBP = 0; %% Guass-Seidel Belief Propagation
useCCCP = 0; %% Convex Concave double loop algorithm
useBADMM = 1; %% Bregman ADMM
for n = [20,50,100]
    for sigma =  [1,2,5] 
        rng('default'); 
        %% generate data
        if Atype == 1 % 2D
            G = gen2Dgrid(n);
            n1 = size(G,1);
            m = nnz(G)/2;
        elseif Atype == 2 % 3D
            G = gen3Dlattice(n);
            n1 = size(G,1);
            m = nnz(G)/2;
        else
            G = random_graph_m(n,ceil(n^2/4)); 
            n1 = size(G,1);
            m = nnz(G)/2;
        end
        C = cell(m,1);
        c = cell(n1,1);
        for i = 1:n1
            c{i} = sigma*randn(r,1);
        end
        [I,J] = find(triu(G));
        for i = 1:m
            C{i} = sigma*randn(r,r);
        end
        %% Jacobi belief Propagation
        if useJBP
            runhist = JBP(G,C,c,T,par);
            ttime = runhist.ttime;
            fval = runhist.fval;
            pfeas = runhist.pfeas;
            iter = runhist.iter;
            fprintf(' \n $n$=%2d & JBP & %3.2e & 0 & %6.7e & %2d & %3.2e \\\\',n1,pfeas,fval,iter,ttime);
        else
            fprintf(' \n $n$=%2d & JBP & - & - & - & -&-  \\\\',n1);
        end
        %% Guass-Seidel belief Propagation
        if useGSBP
            runhist = GSBP(G,C,c,T,par);
            ttime = runhist.ttime;
            fval = runhist.fval;
            pfeas = runhist.pfeas;
            iter = runhist.iter;
            fprintf(' \n $m$=%2d & GSBP & %3.2e & 0 & %6.7e & %2d & %3.2e \\\\',m,pfeas,fval,iter,ttime);
        else
            fprintf(' \n $m$=%2d & GSBP & - & - & - & - &-  \\\\',m);
        end
        %% Bregman ADMM
        if useBADMM
            runhist = BADMM(G,C,c,T,par);
            ttime = runhist.ttime;
            fval = runhist.fval;
            pfeas = runhist.pfeas;
            dfeas = runhist.dfeas;
            iter = runhist.iter;
            fprintf(' \n $\\sigma$=%3.2e & BADMM & %3.2e & %3.2e& %6.7e & %2d & %3.2e \\\\',sigma,pfeas,dfeas,fval,iter,ttime);
        else
            fprintf(' \n $\\sigma$=%3.2e & BADMM & - & - & - & -&-  \\\\',sigma);
        end
        %% Convex Concave procedure
        if useCCCP
            runhist = CCCP(G,C,c,T,par);
            fval = runhist.fval;
            pfeas = runhist.pfeas;
            dfeas = runhist.dfeas;
            ttime = runhist.ttime;
            Initer = runhist.Initer;
            fprintf(' \n  & CCCP & %3.2e & %3.2e & %6.7e & %2d & %3.2e \\\\',pfeas,dfeas,fval,Initer,ttime);
        else
            fprintf(' \n  & CCCP & - & - & - & -& - \\\\');
        end
        fprintf(' \n \\hline');
    end
end

