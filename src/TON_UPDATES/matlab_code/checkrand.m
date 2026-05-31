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

%% initialization
Delta=ones(M,km); % State initialization
pavg=zeros(1,K);
presult=0;
for t=1:K
    %% Initialization
    g=zeros(M,km);
    Ccurr=zeros(1,M);
    Ncurr=0;
    Change=zeros(M,km);
    source=randperm(M,min(N,M));
    i=1;
    while Ncurr<N
        row=source(i);
        columnrand=randperm(km,min(c(row),km));
        for j=1:length(columnrand)
            column=columnrand(j);
              g(row, column)=1;
              Delta(row, column)=1;
              Ccurr(row)=Ccurr(row)+1;
              Ncurr=Ncurr+n(row, column);
              Change(row, column)=1;
        end
        i=i+1;
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
end
presult