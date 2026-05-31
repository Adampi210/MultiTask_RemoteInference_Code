function presult=lowerbound(M,N,km,T,B,K,n,c,w,gamma,p)

%% gain index calculation
subgradientiter1(M, N, T, B, gamma, p, km, w, n, c);
lambdasource=zeros(M, T);
mu=zeros(1, T);
multipliers=load('multipliers.mat', 'lambdasource', 'mu');
lambdasource=multipliers.lambdasource;
mu=multipliers.mu;
asource1(1:M,1:T,1:B, 1:km)=0; % gain index for all source for task 1
%asource2(1:M,1:T,1:B,1:B,1:4)=0; % gain index for all source for task 2
 for m=1:M
        for task=1:km
        lambda=lambdasource(m,:);
        a=valuefunction1(lambda, mu, B, w(m, task)*p(task,:), T, gamma);
        asource1(m,:,:,task)=a;
        end
 end

%% initialization
Delta=ones(M,km); % State initialization
pavg=zeros(1,K);
presult=0;
for t=1:K
    %% Initialization
    g=zeros(M,km);
    gainindex1=zeros(M, km);
    for m=1:M
        for task=1:km
            gainindex1(m, task)=asource1(m,t,Delta(m,task),task);
        end
    end
   for m=1:M
      for j=1:km
         if gainindex1(m,j)>0
             g(m,j)=1;
             Delta(m,j)=1;
         else
             Delta(m,j)=Delta(m,j)+1;
             if Delta(m,j)>B
                 Delta(m,j)=B;
             end
         end
        pavg(t)=pavg(t)+(w(m, j)*p(j,Delta(m,j)))/(km*M)+lambdasource(m,t)*g(m,j)+mu(t)*g(m,j);
      end
   end

    if t<T
    presult=presult+gamma^(t-1)*pavg(t);
    end
end
presult
end