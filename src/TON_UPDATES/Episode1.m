function A=Episode1(asource1, M, T, B, gamma, N, beta, lambdasource, mu, km, w, n, c)
A=zeros(M+1,T);
A(1:M,:)=lambdasource;
A(M+1,:)=mu;
Delta=ones(M,km); % State initialization
for t=1:T
    %% Initialization
    g=zeros(M,km);
    gainindex1=zeros(M, km);
    for m=1:M
        for task=1:km
            gainindex1(m, task)=asource1(m,t,Delta(m,task),task);
        end
    end
   for m=1:M
   for task=1:km
       if gainindex1(m, task)>0;
           g(m, task)=1;
           Delta(m, task)=1;
       else
           g(m, task)=0;
           Delta(m, task)=Delta(m, task)+1;
           if Delta(m, task)>B
               Delta(m, task)=B;
           end
       end
   end
%% Resources used
    Ncurr=0;
    Ccurr=zeros(1,M);
    for m=1:M
        for task=1:km
        Ncurr=Ncurr+g(m, task)*n(m, task);
        Ccurr(m)=Ccurr(m)+g(m, task);
        end
    end

    %% lagrangian update
    for m=1:M
        A(m,t)=A(m, t)+beta*(Ccurr(m)-c(m))*gamma^(t-1); % computation resources
        if A(m, t)<0
          A(m, t)=0;
        end
    end
    A(M+1, t)=A(M+1, t)+beta*gamma^(t-1)*(Ncurr-N); % channel resources
    if A(M+1, t)<0
        A(M+1, t)=0;
    end
end
end
end


        
