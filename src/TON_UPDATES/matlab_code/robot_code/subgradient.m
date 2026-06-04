function A=subgradient(M, N, T, B, gamma, p, km, Delta)
%tic
lambdasource=zeros(1, M);
mu=0;
beta=0.9;
asource1(1:M,1:T,1:B,1:km)=-100; %gain index

for j=1:2000
    for m=1:M
        for task=1:km(m)
        lambda=lambdasource(m);
        a=valuefunction(lambda, mu, B, p(m, task, :), T, gamma);
        asource1(m,:,:,task)=a;
        end
    end
    A=episode(asource1, M, T, B, gamma, N, beta/j, lambdasource, mu, km, Delta);
    lambdasource=A(1:M);
    L(j,:)=lambdasource;
    mu=A(M+1);
    MU(j)=mu;
end
% figure(1)
% plot(L(:,1))
% figure(2)
% plot(L(:,2))
% figure(3)
% plot(MU)
file='multipliers.mat';
save(file,"mu","lambdasource");
%toc
end