function presult=MAF(M,N,km,T,B,gamma,p)

Delta=ones(M,max(km)); % State initialization
pavg=zeros(1,T);
presult=0;
for t=1:T
    %% Initialization
    g=zeros(M,max(km));
    Ccurr=zeros(1,M);
    Ncurr=0;
    Change=zeros(M,max(km));
    D=Delta;
    while max(max(D))>0
      [vr, index]=max(D);
      [vc, column]=max(vr);
      row=index(column);
      n1=Ncurr+1;
      if n1<=N && Ccurr(row)+1<=1
          g(row, column)=1;
          Delta(row, column)=1;
          Ccurr(row)=Ccurr(row)+1;
          Ncurr=n1;
          Change(row, column)=1;
      end
      D(row, column)=0;
    end
    %Change
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
           pavg(t)=pavg(t)+(p(m,j,Delta(m,j)))/(sum(km));
        end
    end
    presult=presult+gamma^(t-1)*pavg(t);
end
presult
end
