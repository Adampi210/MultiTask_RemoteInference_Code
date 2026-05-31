close all;
clear all;
clc;

%% number of sources, channels, bound, and time
M=20;
N=10;
km=2;
B=20;
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
           w(m,j)=0.01;
        end
    end
end
w(1,2)=1;
w(5,1)=1;
K=T;
gamma=0.9;
%% penalty function
delta=1:B;
loss=load('loss.mat', 'p1', 'p2');
p=zeros(km,B);
p1=loss.p1;
p2=loss.p2;
i=1;
for j=1:5:100
    %p(1,i)=100*p1(j)-100*p1(1);
    %p(2,i)=p2(j)-p2(1);
    p(1,i)=p1(j);
    p(2,i)=p2(j);
    i=i+1;
end

channel=2:2:20;
for ch=1:length(channel)
N=channel(ch)
pmgf(ch)=MGF1(M,N,km,T,B,K,n,c,w,gamma,p);
% randpolicy
pmaf(ch)=MAF1(M,N,km,T,B,K,n,c,w,gamma,p);
% MAF
prand(ch)=randpolicy(M,N,km,T,B,K,n,c,w,gamma,p);
end
plot(channel, prand,'ro-', channel, pmaf, 'b*-', channel, pmgf, 'LineWidth',2,'MarkerSize',10)
xlabel('Number of Channels'), ylabel('Discounted Sum of Errors')
legend('Random Policy', 'MAF Policy', 'MGF Policy')
%plot(channel, pmaf, 'b*-', channel, pmgf, 'LineWidth',2,'MarkerSize',10)