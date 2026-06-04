function a=valuefunction(lambda, mu, B, p, T, gamma)
V(1:T+1, 1:B+1)=0;
for i=T:-1:1
        for d=1:B
            Q1=p(d)+gamma*V(i+1, d+1);
            Q2=p(d)+lambda+mu+gamma*V(i+1, 1);
            V(i,d)=min(Q1, Q2);
            a(i, d)=Q1-Q2;
        end 
        V(i,B+1)=V(i,B);
end
end 