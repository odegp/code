function [samptheta, samphyper, SAMPLEX]= ode_ComGP_solve(Data, ODE, GP, Iteration)
%% ODE parameter estimation with common GP settings 

%%%%%%%%%%%%%%%%%% Inputs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Data:
%      Data.y: observations
%      Data.samptime: the sample time

% ODE:
%      ODE.fun: ODE functions
%      ODE.num: the number of the parameters
%      ODE.discrete: the discretized ranges of the parameters
%      ODE.initial: initial values of ODE parameters
%      ODE.prior: prior for ODE parameters 

% GP:
%      GP.fun: GP covariance function: c(t,t')
%      GP.fun_d: GP derivative d{c(t,t')}/dt
%      GP.fun_dd: GP derivative d^2{c(t,t')}/dtdt'
%      GP.num: the number of the GP hyperparameters (we put the noise std to the end of the hyperparameter vector)
%      GP.discrete: the discretized ranges of the GP hyperparameters
%      GP.inital_X: initial mean, std of X, and the number of discretized bins for X
%      GP.inital: initial values of GP hyperparameters
%      GP.prior: prior for GP hyperparameters

% Iter:
%      Iteration.total: the number of the total iterations
%      Iteration.sub:   the number of the sub iterations

%%%%%%%%%%%%%%%%%%% Outputs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% samptheta: samples of ODE parameters
% samphyper: samples of GP hyperparameters
% SAMPLEX: samples of X


%% Load Data and Sampletime
y=Data.y;
samptime=Data.samptime;

TTT=length(samptime);      % number of sampled points
meany=repmat(mean(y),TTT,1);    % observation mean

%% Discratized Grid
%ODE
for i=1:ODE.num
    ODE_par{i}=ODE.discrete{i};
end
%GP
for i=1:GP.num
    GP_par{i}=GP.discrete{i};
end

%% Initialization 
%ODE
odepar=ODE.initial;

%GP
MeanX=GP.initial_X{1};
StdX=GP.initial_X{2};
SX=MeanX;  %%% initial X as the mean of GP

gppar=GP.initial;

%% Gibbs Sampling

for iter=1:Iteration.total
    iter
    %##############  Sampling Parameters #################################   
    for subiter=1:Iteration.sub 
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% Sample GP hyperparameters, fix ODE parameter, noise std and X
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %##### Note before sampling hyperparameters##############
        % Since X and ODE parameters are fixed, dotX are fixed
        for i=1:TTT
            dotx(i,:)=feval(ODE.fun,samptime(i),SX(i,:),odepar)';
        end    
        
        %#################################################################
        
        %%% Sample GP hyperparameters
        for k=1:GP.num-1             
            IT=GP_par{k}; % choose which hyperparameter will be sampled
            
            for j=1:length(IT)   
                gppar(k)=IT(j);
                
                for s_sampind=1:TTT
                    s=samptime(s_sampind);
                    for t_sampind=1:TTT
                        t=samptime(t_sampind);
                        Cxx(s_sampind,t_sampind)=feval(GP.fun,s,t,gppar);  
                        dC(s_sampind,t_sampind)=feval(GP.fun_d,s,t,gppar);                                             
                        ddC(s_sampind,t_sampind)=feval(GP.fun_dd,s,t,gppar);
                    end
                end
                
                Cxx=0.5*(Cxx+Cxx');  
                
                % p(y|dotx)
                Mygxdot=dC'*inv(ddC);
                meandotx=Mygxdot*dotx+meany; % mean

                Cyy=Cxx+gppar(GP.num)^2*eye(TTT); 
                Cygxdot=Cyy-Mygxdot*dC;      % Covariance
                Cygxdot=0.5*(Cygxdot+Cygxdot');  % numerically stable

                p1=1;
                for i=1:size(y,2)
                    p1=p1*mvnpdf(y(:,i)',meandotx(:,i)',Cygxdot);
                end

                % p(x|hyperparameters)
                p2=1;
                for i=1:size(y,2)
                    p2=p2*mvnpdf(SX(:,i)',meany(:,i)',Cxx);  
                end
                
                % Prior
                p3=feval(GP.prior{k},gppar(k));
                pgp(j)=p1*p2*p3;
            end
            
            ind = randsample(length(IT),1,true,pgp);          
            gppar(k)=IT(ind);
            clear IT
            clear pgp
        end        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% Sample noise std, fix GP hyperparameters, ODE parameter and X
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %##### Note before sampling  noise std ###################
        
        % Since X and ODE parameters are fixed, dotX are still fixed as before
        
        % Since GP hyperparameters are fixed now, the covariance matrice are fixed
        
        for s_sampind=1:TTT
            s=samptime(s_sampind);
            for t_sampind=1:TTT
                t=samptime(t_sampind);
                Cxx(s_sampind,t_sampind)=feval(GP.fun,s,t,gppar);  
                dC(s_sampind,t_sampind)=feval(GP.fun_d,s,t,gppar);                                             
                ddC(s_sampind,t_sampind)=feval(GP.fun_dd,s,t,gppar);
            end
        end
        
        % Hence Mean of p(y|dotx) is fixed now
        Mygxdot=dC'*inv(ddC);
        meandotx=Mygxdot*dotx+meany;
        
        % Hence Covaraince of p(y|dotx) is now fixed except sigma^2*I
        Cmiddle=Cxx-Mygxdot*dC;
       
        %#################################################################
                 
        % Sampling Sigma
        IT=GP_par{GP.num};
        for j=1:length(IT)
            gppar(GP.num)=IT(j);
            
            % p(y|dotx)
            Cygxdot=Cmiddle+gppar(GP.num)^2*eye(TTT); % Whole Covariance
            Cygxdot=0.5*(Cygxdot+Cygxdot');
            
            p1=1;
            for i=1:size(y,2)
                p1=p1*mvnpdf(y(:,i)',meandotx(:,i)',Cygxdot);
            end

            % Prior
            p2=feval(GP.prior{GP.num},gppar(GP.num)); 
            psg(j)=p1*p2;
        end

        ind = randsample(length(IT),1,true,psg);
        gppar(GP.num)=IT(ind);
        clear IT
        clear psg
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% Sample ODE parameter, fix GP hyperparameters,  noise std and X
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %##### Note before sampling  ODE parameter ###################      
        
        % Since GP hyperparameters are fixed, covariance matrice are still fixed
              
        % Hence Mygxdot=dC'*inv(ddC) in the Mean of p(y|dotx) is still fixed
               
        % Since noise std is now fixed, Hence Covaraince of p(y|dotx) is fixed
        
        Cygxdot=Cmiddle+gppar(GP.num)^2*eye(TTT);
        Cygxdot=0.5*(Cygxdot+Cygxdot');
        %#################################################
        
        %%% Sampling ODE parameters
        for k=1:ODE.num
            IT=ODE_par{k}; 
            for j=1:length(IT)
                odepar(k)=IT(j);
                
                % p(y|dotx)
                for i=1:TTT
                     dotx(i,:)=feval(ODE.fun,samptime(i),SX(i,:),odepar)';
                     
                end             
                meandotx=Mygxdot*dotx+meany;

                p1=1;
                for i=1:size(y,2)
                    p1=p1*mvnpdf(y(:,i)',meandotx(:,i)',Cygxdot);
                end

                % Prior
                p2=feval(ODE.prior{k},odepar(k));
                
                pode(j)=p1*p2;
            end

            ind = randsample(length(IT),1,true,pode);
            odepar(k)=IT(ind);
            clear IT
            clear pode
        end
          
    end
    
    samptheta(iter,:)=odepar;
    samphyper(iter,:)=gppar;
    
    %############# Sampling X along with dimension ########################
    
    %##### Note before sampling  X ###################      
        
    % Since GP hyperparameters and noise std are fixed,
              
    % Hence Mygxdot=dC'*inv(ddC) in the Mean of p(y|dotx) is fixed
    % Hence Covaraince of p(y|dotx) is fixed
     %#################################################
    
    for subiter=1:Iteration.sub
        
        for Tstep=1:TTT
            
            for DIM=1:size(y,2)
                
                 IT1=linspace(MeanX(Tstep,DIM)-3*StdX(Tstep,DIM),MeanX(Tstep,DIM)+3*StdX(Tstep,DIM),GP.initial_X{3});%
        
                 for j=1:length(IT1)
                  
                     SX(Tstep,DIM)=IT1(j);
                     % p(y|dotx)               
                     
                     for i=1:TTT
                         dotx(i,:)=feval(ODE.fun,samptime(i),SX(i,:),odepar)';
                         
                     end             
                     meandotx=Mygxdot*dotx+meany;
                     
                     p1S=1;
                     for i=1:size(y,2)
                         p1S=p1S*mvnpdf(y(:,i)',meandotx(:,i)',Cygxdot);
                     end
                     
                    % p(x|hyperparameters)
                     p2S=mvnpdf(SX(:,DIM)',meany(:,DIM)',Cxx);
                     px1(j)=p1S*p2S;
                 end
                 ind=randsample(length(IT1),1,true,px1);
                 SX(Tstep,DIM)=IT1(ind);
                 clear IT1
                 clear px1
            end
            
        end
             
    end
    
    SAMPLEX{iter}=SX;
           
end        
        
