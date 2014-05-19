function [MeanX,StdX]=FitGP(Data,GP)
y=Data.y;
samptime=Data.samptime;
TTT=size(y,1);
meany=repmat(mean(y),TTT,1);    % observation mean

for i=1:GP.num
    MI{i}=GP.discrete{i};
    
    vID{1,i}=1:length(GP.discrete{i});
end

nID = numel(vID);
[vID{1:nID}] = ndgrid(vID{:});
ID = reshape(cat(nID,vID{:}),[],nID);

I_NUM=size(ID,1);

for i=1:I_NUM
    
    for j=1:GP.num
        WP(j)=feval(GP.prior{j},MI{j}(ID(i,j)));
    end

    for j=1:GP.num-1
        ppar(j)=MI{j}(ID(i,j));
    end
    
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
        W_y=W_y*mvnpdf(y(:,dimension)',meany(:,dimension)',Cxx+MI{GP.num}(ID(i,GP.num))^2*eye(TTT));
    end        
              
    W_HYP(i)=prod(WP)*W_y;            
                              
end
                
ind_hyp = randsample(I_NUM,200,true,W_HYP);  % 200 samples of hyperarameter vector

for i=1:200
    
     for j=1:GP.num-1
         ppar(j)=MI{j}(ID(ind_hyp(i),j)); 
     end 

     for s_sampind=1:TTT
        s=samptime(s_sampind);
        for t_sampind=1:TTT
            t=samptime(t_sampind);
            Cxx(s_sampind,t_sampind)=feval(GP.fun,s,t,ppar);    
        end                    
     end
     Cxx=0.5*(Cxx+Cxx'); 
     
     ppar(GP.num)=MI{GP.num}(ID(ind_hyp(i),GP.num));  
     

     % mean
     Cxy=Cxx;
     Cyy=Cxx+ppar(GP.num)^2*eye(TTT);
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


for i=1:size(y,2)
    X{i}=zeros(TTT,1);
end

for i=1:200
    for X_NUM=1:20
        UX=SAM_X{i,X_NUM};
        for k=1:size(y,2)
            X{k}=[X{k} UX(:,k)];
        end
    end
end

for i=1:size(y,2)
    MeanX(:,i)=mean(X{i}(:,2:end),2);
    StdX(:,i)=std(X{i}(:,2:end),0,2);
end
