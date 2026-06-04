function presult=MGFReoptimized(M,N,km,T,B,gamma,p)

%% initialization
Delta=ones(M,max(km)); % State initialization
pavg=zeros(1,T);
presult=0;
for t=1:T
    clear asource1
    %Recalculate gain index

%     Delta
    multiplier=subgradient(M, N, T-t+1, B, gamma, p, km, Delta);

    multipliers=load('multipliers.mat', 'lambdasource', 'mu');
    lambdasource=multipliers.lambdasource;
    mu=multipliers.mu;

%      lambdasource=zeros(1, M);
%      mu=0;

    asource1(1:M,1:T-t+1,1:B, 1:max(km))=-100; 
    for m=1:M
        for task=1:km(m)
        lambda=lambdasource(m);
        a=valuefunction(lambda, mu, B, p(m, task, :), T-t+1, gamma);
        asource1(m,:,:,task)=a;
        end
    end

    %% Initialization
    g=-ones(M,max(km))*100;
    gainindex1=zeros(M, max(km));
    for m=1:M
        for task=1:km(m)
            gainindex1(m, task)=asource1(m,1,Delta(m,task),task);
        end
    end
    %gainindex1

    Ccurr=zeros(1,M);
    Ncurr=0;
    Change=zeros(M,max(km));
    while max(max(gainindex1))>=0
      [vr, index]=max(gainindex1);
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
      gainindex1(row, column)=-100;
    end
%     Change

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
    if t<T
    presult=presult+gamma^(t-1)*pavg(t);
    end
    Ncurr;
   Delta;
end
presult
end