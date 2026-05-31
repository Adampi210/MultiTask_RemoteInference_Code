close all;
clear all;
clc;

%% number of sources, channels, bound, and time
M=10;
N=10;
km=2;
B=20;
T=10;
%% channel needed
for m=1:M
for j=1:km
    if j>km/2
       n(m,j)=1;
    else
        n(m, j)=1;
    end
end
end
%% computation resources
for m=1:M
    if m>M/2
        c(m)=1;
    else
        c(m)=1;
    end
end
%% different weight
w=zeros(1,M);
for m=1:M
    for j=1:km
      w(m, j)=m/km;
    end
end
K=T;
gamma=0.9;
%% penalty function
delta=1:B;
% loss=load('loss.mat', 'p1', 'p2');
% p=zeros(km,B);
% p1=loss.p1;
% p2=loss.p2;
% i=1;
% for j=1:5:100
%     p(1,i)=p1(j);
%     p(2,i)=0.01*p2(j);
%     i=i+1;
% end
p(1,1:B)=log(1:B);
p(2,1:B)=1:B;
%% gain index calculation
subgradientiter1(M, N, T, B, gamma, p, km, w, n, c);
lambdasource=zeros(M, T);
mu=zeros(1, T);
multipliers=load('multipliers.mat', 'lambdasource', 'mu');
lambdasource=multipliers.lambdasource;
mu=multipliers.mu
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
    gainindex1;
    Ccurr=zeros(1,M);
    Ncurr=0;
    Change=zeros(M,km);
    while max(max(gainindex1))>0
      [vr, index]=max(gainindex1);
      [vc, column]=max(vr);
      row=index(column);
      n1=Ncurr+n(row, column);
      if n1<=N && Ccurr(row)+1<=c(row)
          g(row, column)=1;
          Delta(row, column)=1;
          Ccurr(row)=Ccurr(row)+1;
          Ncurr=n1;
          Change(row, column)=1;
      end
      gainindex1(row, column)=0;
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
    if t<T
    presult=presult+gamma^(t-1)*pavg(t);
    end
  Delta
end
presult



