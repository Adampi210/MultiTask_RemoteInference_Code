function A=episode(asource1, M, T, B, gamma, N, beta, lambdasource, mu, km, Delta)
A=zeros(M+1);
A(1:M)=lambdasource;
A(M+1)=mu;
Ncurr=zeros(1,T);
Ccurr=zeros(T,M);
for t=1:T
    %% Initialization
    g=zeros(M,max(km));
    gainindex1=zeros(M, max(km));
    for m=1:M
        for task=1:km(m)
            gainindex1(m, task)=asource1(m,t,Delta(m,task),task);
        end
    end
   for m=1:M
   for task=1:km(m)
       if gainindex1(m, task)>0
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
   end
%% Resources used  
    for m=1:M
        for task=1:km(m)
        Ncurr(t)=Ncurr(t)+g(m, task);
        Ccurr(t, m)=Ccurr(t, m)+g(m, task);
        end
        Ccurr(t, m)=gamma^(t-1)*Ccurr(t, m);
    end
    Ncurr(t)=gamma^(t-1)*Ncurr(t);
end
    %% lagrangian update
    r=(1-gamma^T)/(1-gamma);

    for m=1:M
        A(m)=max(A(m)+beta*(sum(Ccurr(:,m))-r), 0); % computation resources
    end
    A(M+1)=max(A(M+1)+beta*(sum(Ncurr)-N*r), 0); % channel resources
    if A(M+1)<0
        A(M+1)=0;
    end
end