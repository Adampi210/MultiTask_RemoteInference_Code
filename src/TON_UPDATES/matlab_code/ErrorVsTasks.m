close all;
clear all;
clc;
B=20;
delta=1:B;
for j=1:15
  if rem(j,3)==0
     p(j,1:B)=1:B;
  elseif rem(j,3)==1
     p(j,1:B)=10*log(1:B);
  else
      p(j,1:B)=exp(0.5*(1:B));
  end
end

%% number of sources, channels, bound, and time
M=20;
N=10;

T=100;

%w(1,2)=1;
%w(5,1)=1;
Tasks=3:3:15;
K=T;
gamma=0.9;
for ch=1:length(Tasks)
    km=Tasks(ch)
    %% channel needed
n=ones(M,km);
%% computation resources
c=ones(M,km)*2;
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
plot(Tasks, prand,'ro-', Tasks, pmaf, 'b*-', Tasks, pmgf, 'LineWidth',2,'MarkerSize',10)
xlabel('Number of Tasks'), ylabel('Discounted Sum of Errors')
legend('Random Policy', 'MAF Policy', 'MGF Policy')
%plot(channel, pmaf, 'b*-', channel, pmgf, 'LineWidth',2,'MarkerSize',10)