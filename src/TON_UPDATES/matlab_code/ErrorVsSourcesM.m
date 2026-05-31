close all;
clear all;
clc;

%% number of sources, channels, bound, and time
M=40;
N=10;
km=9;
B=20;
T=100;

%w(1,2)=1;
%w(5,1)=1;
K=T;
gamma=0.9;
%% penalty function
for j=1:km
    if rem(j,3)==0
     p(j,1:B)=1:B;
  elseif rem(j,3)==1
     p(j,1:B)=10*log(1:B);
  else
      p(j,1:B)=exp(0.5*(1:B));
  end
end

sources=3:3:21;
for ch=1:length(sources)
    M=sources(ch)
    %% channel needed
n=ones(M,km);
%% computation resources
c=ones(M,km)*2;
%% different weight
w=zeros(1,M);
for m=1:M
        if m>0.5*M
          w(m,j)=0.01;
        else
           w(m,j)=1;
        end
end
pmgf(ch)=MGF1(M,N,km,T,B,K,n,c,w,gamma,p);
% randpolicy
pmaf(ch)=MAF1(M,N,km,T,B,K,n,c,w,gamma,p);
% MAF
prand(ch)=randpolicy(M,N,km,T,B,K,n,c,w,gamma,p);
end
plot(sources, prand,'ro-', sources, pmaf, 'b*-', sources, pmgf, 'LineWidth',2,'MarkerSize',10)
xlabel('Number of Sources'), ylabel('Discounted Sum of Avg. Error')
legend('Random Policy', 'MAF Policy', 'MGF Policy')
