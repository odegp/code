clear all; 
close all;

%%  Integration Time and Sample Time

T=2;                       % real time [0,2]

del=0.01;                  % integration interval
truetime=0:del:T;          % true times
Tindex=length(truetime);   % index time


Tdel=0.2;                  % sample interval- In the paper [F. Dondelinger et al, AISTATS 2013], it is mentioned that the sample rate is 0.25. But in the experiment we can see that there are 11 data points, which means that the sample rate is 0.2
samptime=0:Tdel:T;         % sample times
TTT=length(samptime);      % number of sampled points

itrue=round(samptime./del+ones(1,TTT)); % Index of sample time in the true time

%% Ground Truth Generation by numerical integration

xx(1,:) = [5 3];              % Initial values of latent states
theta_true=[2 1 4 1];         % Truth parameters in ODE

%%%%% Ground Truth 

[ToutX,OutX]=ode45(@(t,x) LV(t,x,theta_true), truetime, xx(1,:));
xx=OutX;
x=xx(itrue,:);

%% Observation Generation

sigma2=0.25;                    % noise variance
sigma=sqrt(sigma2);

y=x+sigma*randn(size(x));     % add noise


%% Data:
Data.y=y; % observations
Data.samptime=samptime; % the sample time of observations

%% ODE:
ODE.fun=@LV; % ODE functions
ODE.num=4;   % the number of the parameters
ODE.discrete{1}=1.5:0.1:2.5;  %the discretized ranges of the parameters
ODE.discrete{2}=0.5:0.1:1.5;
ODE.discrete{3}=3.5:0.1:4.5;
ODE.discrete{4}=0.5:0.1:1.5;
ODE.initial=[1.5 0.5 3.5 0.5]; %initial values of ODE parameters
ODE.prior{1}=@(x) gampdf(x,4,0.5);% prior for ODE parameters 
ODE.prior{2}=@(x) gampdf(x,4,0.5);
ODE.prior{3}=@(x) gampdf(x,4,0.5);
ODE.prior{4}=@(x) gampdf(x,4,0.5);

%% X0:
X0.indicator=1; % Indicate whether X0 prior is separately specified. If specified, X0.indicator=1
if X0.indicator==1
   X0.discrete{1}=linspace(1,10,20);% the discretized ranges of x0
   X0.discrete{2}=linspace(1,10,20);
   X0.initial=[5 3]; %initial values of x0;
   X0.prior{1}=@(x) unifpdf(x,0,10); % prior of x0
   X0.prior{2}=@(x) unifpdf(x,0,10);
end

%% GP:
%%%%%%%%%%%%%%% RBF Kernel %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
GP.fun=@GPcov; % GP covariance function: c(t,t')
GP.fun_d=@dGPcov; % GP derivative d{c(t,t')}/dt
GP.fun_dd=@ddGPcov; %GP derivative d^2{c(t,t')}/dtdt'
GP.num=3; % the number of the GP hyperparameters (we put the noise std to the end of the hyperparameter vector)
GP.discrete{1}=0.1:0.1:1; %(pref) % the discretized ranges of the GP hyperparameters
GP.discrete{2}=5:5:50;    %(lengthscale)
GP.discrete{GP.num}=0.1:0.1:1; %(sigma)
GP.prior{1}=@(x) unifpdf(x,0,100);%prior for GP hyperparameters
GP.prior{2}=@(x) unifpdf(x,0,100);
GP.prior{GP.num}=@(x) gampdf(x,1,1);
GP.initial=[1 10 0.5]; %initial values of GP hyperparameters

%%%%%%%%%%%%%%% Sigmoid Kernel %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GP.fun=@GPcovSIG; % GP covariance function: c(t,t')
% GP.fun_d=@dGPcovSIG; % GP derivative d{c(t,t')}/dt
% GP.fun_dd=@ddGPcovSIG; %GP derivative d^2{c(t,t')}/dtdt'
% GP.num=4; % the number of the GP hyperparameters (we put the noise std to the end of the hyperparameter vector)
% GP.discrete{1}=1:2:20;%    % pref % the discretized ranges of the GP hyperparameters
% GP.discrete{2}=1:2:20;%    % a
% GP.discrete{3}=1:2:20;%    % b
% GP.discrete{GP.num}=0.1:0.1:1;% % sigma
% GP.prior{1}=@(x) unifpdf(x,0,100);%prior for GP hyperparameters
% GP.prior{2}=@(x) unifpdf(x,0,100);
% GP.prior{3}=@(x) unifpdf(x,0,100);
% GP.prior{GP.num}=@(x) gampdf(x,1,1);
% GP.initial=[10 10 10 0.5]; %initial values of GP hyperparameters

%%%%%%%%%% Initialization of X by using GP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('--------------Fit the data by GP to obtain the initial X-------------------');
    
[MeanX,StdX]=FitGP(Data,GP);

disp('--------------GP Fit is done, Plot GP Fit results-------------------');

disp('--------------Press /Enter/ to start sampling-------------------');
for dimension=1:2
    figure
    plot(truetime(itrue),MeanX(:,dimension),'r-');
    hold on
    plot(truetime(itrue),x(:,dimension),'o-')
    hold on
    plot(truetime(itrue),MeanX(:,dimension)-3*StdX(:,dimension),'r-')
    hold on
    plot(truetime(itrue),MeanX(:,dimension)+3*StdX(:,dimension),'r-')
    hold on
    plot(truetime(itrue),y(:,dimension),'*')
end

pause

GP.initial_X{1}=MeanX; % initial mean of X
GP.initial_X{2}=StdX; % initial std of X
GP.initial_X{3}=20; % the number of discretized bins for X



%% Iter:
Iteration.total=100; %the number of the total iterations
Iteration.sub=10;   %the number of the sub iterations


%% Solver
disp('--------------Start Sampling-------------------');
if X0.indicator==1
    [samptheta, samphyper, samplex0, SAMPLEX]= ode_ComGP_solve_X0(Data, ODE, GP, Iteration, X0);
else
    [samptheta, samphyper, SAMPLEX]= ode_ComGP_solve(Data, ODE, GP, Iteration);
end


disp('--------------Sampling is done-------------------');

%% Plot

disp('---------------Estimated ODE Parameters [alpha, beta, gamma, delta] (Mean and Std)----------------------');

MUtheta=mean(samptheta)
STDtheta=std(samptheta)


%% Reproduce True Curve by ODE parameters
DM=size(y,2);

for i=1:Iteration.total 
True(1,:,i) = xx(1,:);              % Initial values of latent states
[ToutX,OutX]=ode45(@(t,x) LV(t,x,samptheta(i,:)), truetime, True(1,:,i)); 
True(1:Tindex,1:DM,i)=OutX;
end

for i=1:DM
    figure
    RTrue=reshape(True(:,i,:),Tindex,Iteration.total);
    MT=mean(RTrue,2);
    ST=std(RTrue')';
    errorbar(truetime, MT, ST, 'g.' )
    hold on
    plot(truetime,xx(:,i),'.')
    hold on
    plot(samptime,y(:,i),'r*')
    
    legend('Reconstruction with ODE Parameter Samples (Mean+/-Std)','Ground Truth', 'Observations')
end




