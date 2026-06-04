close all;
clear all;
clc;


 
channel=4; %number of channel or M
class=2;
k=10;
user=class*k; %number of user 
T=50; %total number of episodes

i=1; 
for H=10:10:100 % time horizon

P=zeros(user, T); % Probability of successful transmission of users
w=zeros(user,1); % variance of signal values
iteration=50;
for j=1:iteration
j
for n=1:user
        if n>user/2
            w(n)=0.5;
            P(n, 1)=1;
        else
            w(n)=0.9;
            P(n, 1)=0.1;
        end
end
epsilon=0.05;
errorp=0.1;
for t=2:T
    if rand<0.9
        errorp=min(errorp+epsilon, 1);
    else
        errorp=max(errorp-epsilon, 0);
    end
    for n=1:user
        if n>user/2
            P(n, t)=1;
        else
            P(n, t)=errorp;
        end
    end
 end   


gamma=0.99; % discount factor 

RWhittle(j)=WhittleOracle(user, channel, P, w, T, H, gamma, epsilon);
win=1;
R(j)=OurPolicy(user, channel, P, w, T, H, gamma, epsilon, win);
RUC(j)=UCWhittle(user, channel, P, w, T, H, gamma, epsilon);
Runi(j)=Uniform(user, channel, P, w, T, H, gamma, epsilon);
RWIQL(j)=WIQL(user, channel, P, w, T, H, gamma, epsilon);
end
R=abs(RWhittle-R);
RUC=abs(RWhittle-RUC);
Runi=abs(RWhittle-Runi);
RWIQL=abs(RWhittle-RWIQL);


Regret(i)=mean(Regretiter);
Regretconfidence(i)=1.96*std(Regretiter)/sqrt(iteration);
RegretUC(i)=mean(RegretUCiter);
RegretconfidenceUC(i)=1.96*std(RegretUCiter)/sqrt(iteration);
RegretWIQL(i)=mean(RegretWIQLiter);
RegretconfidenceWIQL(i)=1.96*std(RegretWIQLiter)/sqrt(iteration);
Regretuni(i)=mean(Regretuniiter);
Regretconfidenceuni(i)=1.96*std(Regretuniiter)/sqrt(iteration);
i=i+1;
end
figure(1)
semilogy(10:10:100, Regretuni, 'b-.',10:10:100, RegretUC,'r--',10:10:100, RegretWIQL, 'm--', 10:10:100, Regret,'k-');
xlabel('Time Horizon'), ylabel('Reg(T)')
legend('Random', 'UCWhittle','WIQL', 'OurPolicy')
