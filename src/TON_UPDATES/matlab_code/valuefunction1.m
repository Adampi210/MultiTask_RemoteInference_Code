function a=valuefunction1(lambda, mu, B, p, T, gamma)
V(1:T+1, 1:B+1)=0;
for i=T:-1:1
        for d=1:B
            Q(1)=p(d)+gamma*V(i+1, d+1);
            Q(2)=p(d)+lambda(i)+mu(i)+gamma*V(i+1, 1);
            V(i,d)=min(Q);
            a(i, d)=Q(1)-Q(2);
        end 
        V(i,B+1)=V(i,B);
end
end        

