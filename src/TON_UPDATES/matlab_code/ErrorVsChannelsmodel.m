close all;
clear all;
clc;
B=20;
delta=1:B;
km=9;
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
M=20;
N=10;
T=100;
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


%w(1,2)=1;
%w(5,1)=1;
channel=2:2:20;
K=T;
gamma=0.9;
for ch=1:length(channel)
N=channel(ch)
pmgf(ch)=MGF1(M,N,km,T,B,K,n,c,w,gamma,p);
% randpolicy
pmaf(ch)=MAF1(M,N,km,T,B,K,n,c,w,gamma,p);
% MAF
prand(ch)=randpolicy(M,N,km,T,B,K,n,c,w,gamma,p);
end
plot(channel, prand,'ro-', channel, pmaf, 'b*-', channel, pmgf, 'LineWidth',2,'MarkerSize',10)
xlabel('channel'), ylabel('Discounted Sum of Errors')
legend('Random Policy', 'MAF Policy', 'MGF Policy')