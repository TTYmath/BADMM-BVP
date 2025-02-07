%% Test sensor network localization 
clear all 
close all 
addpath('./Tools'); 
addpath('./Solvers/'); 
rng('default'); 
%% parameters
par.maxiter = 10000;
par.tol = 1e-4;
par.verbose = 1;
par.maxtime = 3600; 
na = 4; % number of anchors 
t = 10; % gridlevel, r = t^2
T = 1;
plotyes = 1;
%% solvers
useJBP = 1; %% Jacobian Belief Propagation
useGSBP = 1; %% Guass-Seidel Belief Propagation
useCCCP = 0; %% Convex Concave double loop algorithm
useBADMM = 1; %% Bregman ADMM
pk = 1;
BADMM_data = cell(6,1);
JBP_data = cell(6,1);
GSBP_data = cell(6,1);
CCCP_data = cell(6,1);
for n = 100 
    for R = 0.2 %[0.3,0.5,0.6] %[0.1,0.2]  
        for sig = 0.02 %[0.005,0.01,0.02] %[0.2,0.3] %[0.02,0.05,0.1] % noise level
            rng('default');
            %[G,C,c,Sen,Anc,gnods] = genSNLdata_outlier(sig,R,n,na,t,0.05); % with outlier noise
            [G,C,c,Sen,Anc,gnods] = genSNLdata(sig,R,n,na,t); % no outlier noise
            if ~isempty(find(sum(G,2)==0, 1))
                continue;
            end
            m = nnz(G)/2;
            %% Jacobi belief Propagation
            if useJBP
                runhist = JBP(G,C,c,T,par);
                ttime = runhist.ttime;
                fval = runhist.fval;
                pfeas = runhist.pfeas;
                iter = runhist.iter;
                q = runhist.q;
                JBP_data{pk} = q;
                fprintf(' \n $n$=%2d & JBP & %3.2e & 0 & %6.7e & %2d & %3.2e \\\\',n,pfeas,fval,iter,ttime);
                if plotyes == 1     
                    pSen = q'*gnods;
                    RMSD = norm(pSen-Sen,'fro')/sqrt(n);
                    subplot(2,2,1);
                    plot(Sen(:,1),Sen(:,2),'r*');
                    hold on;
                    plot(Anc(:,1),Anc(:,2),'kp','MarkerSize',12);
                    plot(pSen(:,1),pSen(:,2),'b+');
                    plot([Sen(:,1),pSen(:,1)]',[Sen(:,2),pSen(:,2)]','g-');
                    title(['JBP, RMSD=',num2str(RMSD)]);
                end
            else
                fprintf(' \n $n$=%2d & JBP & - & - & - & -&-  \\\\',n);
            end
            %% Guass-Seidel belief Propagation
            if useGSBP
                runhist = GSBP(G,C,c,T,par);
                ttime = runhist.ttime;
                fval = runhist.fval;
                pfeas = runhist.pfeas;
                iter = runhist.iter;
                q1 = runhist.q;
                r = length(q1{1});
                q = zeros(r,n);
                for k = 1:n
                    q(:,k) = q1{k};
                end
                GSBP_data{pk} = q;
                fprintf(' \n $m$=%2d & GSBP & %3.2e & 0 & %6.7e & %2d & %3.2e \\\\',m,pfeas,fval,iter,ttime);
                if plotyes == 1
                    pSen = q'*gnods;
                    RMSD = norm(pSen-Sen,'fro')/sqrt(n);
                    subplot(2,2,2);
                    plot(Sen(:,1),Sen(:,2),'r*');
                    hold on;
                    plot(Anc(:,1),Anc(:,2),'kp','MarkerSize',12);
                    plot(pSen(:,1),pSen(:,2),'b+');
                    plot([Sen(:,1),pSen(:,1)]',[Sen(:,2),pSen(:,2)]','g-');
                    title(['GSBP, RMSD=',num2str(RMSD)]);
                end
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
                q = runhist.q;
                BADMM_data{pk} = q;
                fprintf(' \n $\\sigma$=%3.2e & BADMM & %3.2e & %3.2e& %6.7e & %2d & %3.2e \\\\',sig,pfeas,dfeas,fval,iter,ttime);
                if plotyes == 1
                    pSen = q'*gnods;
                    RMSD = norm(pSen-Sen,'fro')/sqrt(n);
                    subplot(2,2,3); 
                    plot(Sen(:,1),Sen(:,2),'r*');
                    hold on;
                    plot(Anc(:,1),Anc(:,2),'kp','MarkerSize',12);
                    plot(pSen(:,1),pSen(:,2),'b+');
                    plot([Sen(:,1),pSen(:,1)]',[Sen(:,2),pSen(:,2)]','g-');
                    title(['BADMM, RMSD=',num2str(RMSD)]);
                end
            else
                fprintf(' \n $\\sigma$=%3.2e & BADMM & - & - & - & -&-  \\\\',sig);
            end
            %% Convex Concave procedure
            if useCCCP
                runhist = CCCP(G,C,c,T,par);
                fval = runhist.fval;
                pfeas = runhist.pfeas;
                dfeas = runhist.dfeas;
                ttime = runhist.ttime;
                Initer = runhist.Initer;
                q1 = runhist.q;
                r = length(q1{1});
                q = zeros(r,n);
                for k = 1:n
                    q(:,k) = q1{k};
                end 
                CCCP_data{pk} = q;
                fprintf(' \n  $R$=%3.2e& CCCP & %3.2e & %3.2e & %6.7e & %2d & %3.2e \\\\',R,pfeas,dfeas,fval,Initer,ttime);
                if plotyes == 1
                    pSen = q'*gnods;
                    RMSD = norm(pSen-Sen,'fro')/sqrt(n);
                    subplot(2,2,4); 
                    plot(Sen(:,1),Sen(:,2),'r*');
                    hold on;
                    plot(Anc(:,1),Anc(:,2),'kp','MarkerSize',12);
                    plot(pSen(:,1),pSen(:,2),'b+');
                    plot([Sen(:,1),pSen(:,1)]',[Sen(:,2),pSen(:,2)]','g-');
                    title(['CCCP, RMSD=',num2str(RMSD)]); 
                end
            else
                fprintf(' \n  $R$=%3.2e& CCCP & - & - & - & -& - \\\\',R);
            end
            fprintf(' \n \\hline');
            pk = pk+1;
        end
    end
end
%save('SNL.mat','BADMM_data','CCCP_data','JBP_data','GSBP_data');









