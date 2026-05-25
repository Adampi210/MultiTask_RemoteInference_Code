function A=subgradientiter1(M, N, T, B, gamma, p, km, w, n, c)
lambdasource=zeros(M, T);
mu=zeros(1, T);
beta=0.9;
asource1(1:M,1:T,1:B,1:km)=0; % gain index for all source for task 1
% if N==2
%     Titer=1000000;
% else
%     Titer=10000;
% end
Titer=10000;
for j=1:10000
    for m=1:M
        for task=1:km
        lambda=lambdasource(m,:);
        a=valuefunction1(lambda, mu, B, w(m, task)*p(task,:), T, gamma);
        asource1(m,:,:,task)=a;
        end
    end
    A=Episode1(asource1, M, T, B, gamma, N, beta/j, lambdasource, mu, km, w, n, c);
    lambdasource=A(1:M,:);
    mu=A(M+1,:);
end
file='multipliers.mat';
save(file,"mu","lambdasource");
end
        
        





       



