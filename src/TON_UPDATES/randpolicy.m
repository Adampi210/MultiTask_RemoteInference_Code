function presult=randpolicy(M,N,km,T,B,K,n,c,w,gamma,p)
presult=0;
Titer=100;
presultiter=zeros(1,Titer);
for iter=1:Titer
    Delta=ones(M,km); % State initialization
    pavg=zeros(1,K);
for t=1:K
    %% Initialization
    g=zeros(M,km);
    Change=zeros(M,km);
    source=randperm(M,min(N,M));
    i=1;
    for i=1:length(source)
           row=source(i);
            column=randperm(km,1);
            g(row, column)=1;
            Delta(row, column)=1;
            Change(row, column)=1;
    end

    %% update AoI
    for m=1:M
        for j=1:km
            if Change(m,j)==0
                if Delta(m, j)+1>B
                    Delta(m, j)=B;
                else
                    Delta(m, j)=Delta(m, j)+1;
                end
            end
           pavg(t)=pavg(t)+(w(m, j)*p(j,Delta(m,j)))/(km*M);
        end
    end
   
    if t<T
    presultiter(iter)=presultiter(iter)+gamma^(t-1)*pavg(t);
    end
end
end
presult=mean(presultiter)
end