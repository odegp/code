function [MeanX,StdX]=fitGP(y,GP,samptime)

TTT=size(y,1);
meany=repmat(mean(y),TTT,1);    % observation mean

for i=1:GP.num
    MI(i,:)=GP.discrete{i};
end

I_NUM=1;

for i=1:length(GP.discrete{1}) %pref
    for j=1:length(GP.discrete{2}) %lengthscale
        for k=1:length(GP.discrete{3}) %sigma
            
                ID(I_NUM,:)=[i,j,k];
                
                W_pref=feval(GP.prior{1},MI(1,i));
                
                W_l=feval(GP.prior{2},MI(2,j));
                
                W_sigma=feval(GP.prior{3},MI(3,k));
                
                
                ppar(1)=MI(1,i); 
                ppar(2)=MI(2,j);  

                for s_sampind=1:TTT
                    s=samptime(s_sampind);
                    for t_sampind=1:TTT
                        t=samptime(t_sampind);
                        Cxx(s_sampind,t_sampind)=feval(GP.fun,s,t,ppar);   
                    end                    
                end
                Cxx=0.5*(Cxx+Cxx'); 
                
              
                
                W_y=1;
                for dimension=1:size(y,2)
                    W_y=W_y*mvnpdf(y(:,dimension)',meany(:,dimension)',Cxx+MI(3,k)^2*eye(TTT));
                    % This is not well coded since it assumes that
                    % GP.discrete{3} contains the observation noise. We
                    % need to recode this so that it will work for any
                    % covariance function. I would suggest to use a
                    % separate GP.noisediscrete variable.
                    
                end
                
                W_HYP(I_NUM)=W_pref*W_l*W_sigma*W_y;
                
                I_NUM=I_NUM+1;
         
        end
    end
end
                
ind_hyp = randsample(I_NUM-1,200,true,W_HYP);  % 200 samples of hyperarameter vector

for i=1:200
    
     ppar(1)=MI(1,ID(ind_hyp(i),1)); 
     ppar(2)=MI(2,ID(ind_hyp(i),2));  


     for s_sampind=1:TTT
        s=samptime(s_sampind);
        for t_sampind=1:TTT
            t=samptime(t_sampind);
            Cxx(s_sampind,t_sampind)=feval(GP.fun,s,t,ppar);    
        end                    
     end
     Cxx=0.5*(Cxx+Cxx'); 
     
     ppar(3)=MI(3,ID(ind_hyp(i),3));   % db: similar issues here. This code is not generic

     % mean
     Cxy=Cxx;
     Cyy=Cxx+ppar(3)^2*eye(TTT);
     Mxgy=Cxy*inv(Cyy); 
     MUX=meany+Mxgy*(y-meany);
     % covariance
     Cxgy=Cxx-Cxy*inv(Cyy)*Cxy'; 
     Cxgy=0.5*(Cxgy+Cxgy'); 
     
     for X_NUM=1:20 % for each hyperparameter vector, draw 20 samples of X
     
        for dimension=1:size(y,2)
            sampx(:,dimension)=(mvnrnd(MUX(:,dimension)',Cxgy,1))';
        end
        
        SAM_X{i,X_NUM}=sampx;
     end
     
end

X_1=zeros(TTT,1);
X_2=zeros(TTT,1);

for i=1:200
    for X_NUM=1:20
        UX=SAM_X{i,X_NUM};
        X_1=[X_1 UX(:,1)];
        X_2=[X_2 UX(:,2)];
    end
end

X_1=X_1(:,2:end);
X_2=X_2(:,2:end);


MeanX=[mean(X_1,2) mean(X_2,2)]; % db: This is not generic since it assumes the observations are two dimensional
StdX=[std(X_1,0,2) std(X_2,0,2)];
