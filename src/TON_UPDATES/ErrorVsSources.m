close all;
clear all;
clc;
B=20;
delta=1:B; %AoI value
km=9; %number of tasks

%loss=load('loss.mat', 'p1', 'p2');

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

%% number of sources, channels, bound, and time
N=10; %number of channels
T=100; %Total Time
%% channel needed
n=ones(M,km); 
%% computation resources
c=ones(M,km)*2;

sources=2:2:20;
K=T;
gamma=0.9; % discount factor
for ch=1:length(sources)
    
M=sources(ch)

%% different weight
w=zeros(1,M);
for m=1:M
    for j=1:km
        if m>M/2
          w(m,j)=0.01;
        else
           w(m,j)=1;
        end
    end
end

pmgf(ch)=MGF1(M,N,km,T,B,K,n,c,w,gamma,p);
% randpolicy
pmaf(ch)=MAF1(M,N,km,T,B,K,n,c,w,gamma,p);
% MAF
prand(ch)=randpolicy(M,N,km,T,B,K,n,c,w,gamma,p);
end
plot(sources, prand,'ro-', sources, pmaf, 'b*-', sources, pmgf, 'LineWidth',2,'MarkerSize',10)
xlabel('Number of Sources'), ylabel('Discounted Sum of Errors')
legend('Random Policy', 'MAF Policy', 'MGF Policy')
