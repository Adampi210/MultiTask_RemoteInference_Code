function presult=MAF1(M,N,km,T,B,K,n,c,w,gamma,p)

Delta=ones(M,km); % State (AoI) initialization
pavg=zeros(1,K);
presult=0;
for t=1:K
    %% Initialization
    g=zeros(M,km);
    Ccurr=zeros(1,M);
    Ncurr=0;
    Change=zeros(M,km);
    D=Delta;

    %% scheduling decision 
    while max(max(D))>0
      [vr, index]=max(D);
      [vc, column]=max(vr);
      row=index(column);
      n1=Ncurr+n(row, column);
      if n1<=N && Ccurr(row)+1<=c(row)
          g(row, column)=1;
          Delta(row, column)=1;
          Ccurr(row)=Ccurr(row)+1;
          Ncurr=n1;
          Change(row, column)=1; %scheduling decision 
      end
      D(row, column)=0;
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
    Ncurr;
    Delta;
    if t<T
    presult=presult+gamma^(t-1)*pavg(t);
    end
end
presult
end