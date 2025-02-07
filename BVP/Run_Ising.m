%% Testing Ising model
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
T = 1;   
Atype = 2; % 1, 1D; 2, 2D
%% solvers 
useJBP = 0; %% Jacobi BP 
useGSBP = 0; %% Gauss-Seidel BP
useBADMM = 1; %% Bregman ADMM 
for n = 10 %[10,50]
    for T = 1 %[1,0.1,0.01]
        for hx = 1.05 % 1.05
            for hz = 0.5 % 0.5
                for J = 1 % 1
                    rng('default');
                    %% generate data
                    if Atype == 1
                        G = sparse((1:n-1)',(2:n)',ones(n-1,1),n,n);
                        G = G+G';
                    else
                        G = gen2Dgrid(n);
                    end
                    n1 = size(G,1);
                    m1 = nnz(G)/2;
                    C = cell(m1,1);
                    c = cell(n1,1);
                    cc = hx*[0,1;1,0]-hz*[1,0;0,-1];
                    CC = -J*kron([1,0;0,-1],[1,0;0,-1]);
                    for i = 1:n1
                        c{i} = cc/T;
                    end
                    for i = 1:m1
                        C{i} = CC/T;
                    end
                    %% Jacobi Quantum belief Propagation
                    if useJBP
                        runhist = QT_JBP(G,C,c,par);
                        ttime = runhist.ttime;
                        fval = runhist.fval;
                        pfeas = runhist.pfeas;
                        iter = runhist.iter;
                        if ~isnan(fval)
                            fprintf(' \n $n$=%2d & JBP & %3.2e & 0 & %6.7e & %2d & %3.2e \\\\',n1,pfeas,fval,iter,ttime);
                        else
                            fprintf(' \n $n$=%2d & JBP & - & - & - & -&-  \\\\',n1);
                        end
                    else
                        fprintf(' \n $n$=%2d & JBP & - & - & - & -&-  \\\\',n1);
                    end
                    %% Gauss-Seidel Quantum belief Propagation
                    if useGSBP
                        runhist = QT_GSBP(G,C,c,par);
                        ttime = runhist.ttime;
                        fval = runhist.fval;
                        pfeas = runhist.pfeas;
                        iter = runhist.iter;
                        if~isnan(fval)
                            fprintf(' \n $m$=%2d & GSBP & %3.2e & 0 & %6.7e & %2d & %3.2e \\\\',m1,pfeas,fval,iter,ttime);
                        else
                            fprintf(' \n $m$=%2d & GSBP & - & - & - & -&-  \\\\',m1);
                        end
                    else
                        fprintf(' \n $m$=%2d & GSBP & - & - & - & -&-  \\\\',m1);
                    end
                    %% Bregman ADMM
                    if useBADMM
                        par.maxiter = 10000;
                        runhist = QT_BADMM(G,C,c,par);
                        ttime = runhist.ttime;
                        fval = runhist.fval;
                        pfeas = runhist.pfeas;
                        dfeas = runhist.dfeas;
                        iter = runhist.iter;
                        q = runhist.q;
                        fprintf(' \n $T$=%1.2f & BADMM & %3.2e & %3.2e& %6.7e & %2d & %3.2e \\\\',T,pfeas,dfeas,fval,iter,ttime);
                    else
                        fprintf(' \n $T$=%1.2f & BADMM & - & - & - \\\\',T);
                    end
                    fprintf(' \n \\hline');
                end
            end
        end
    end
end