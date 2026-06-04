function presult=randpolicy(M,N,km,T,B,gamma,p)
presult=0;
Titer=100;
presultiter=zeros(1,Titer);
for iter=1:Titer
    Delta=ones(M,max(km)); % State initialization
    pavg=zeros(1,T);
for t=1:T
    %% Initialization
    g=zeros(M,max(km));
    Change=zeros(M,max(km));
    source=randperm(M,min(N,M));
    i=1;
    for i=1:length(source)
           row=source(i);
            column=randperm(km(row),1);
            g(row, column)=1;
            Delta(row, column)=1;
            Change(row, column)=1;
    end

    %% update AoI
    for m=1:M
        for j=1:km(m)
            if Change(m,j)==0
                if Delta(m, j)+1>B
                    Delta(m, j)=B;
                else
                    Delta(m, j)=Delta(m, j)+1;
                end
            end
           pavg(t)=pavg(t)+(p(m, j,Delta(m,j)))/(sum(km));
        end
    end
   
    if t<T
    presultiter(iter)=presultiter(iter)+gamma^(t-1)*pavg(t);
    end
end
end
presult=mean(presultiter)
end