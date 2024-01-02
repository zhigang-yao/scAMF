function mi = calMI(idx,label)

n = max(label);
idx = idx(:); label = label(:);

Perf=zeros(n);
idx = idx(:); label = label(:);

P1= zeros(n,1);P2= zeros(n,1);
for i=1:length(idx)
    Perf(idx(i),label(i)) = Perf(idx(i),label(i))+1;
    P1(idx(i)) = P1(idx(i))+1;
    P2(label(i)) = P2(label(i))+1;
end

Perf=Perf/length(idx);
P1=P1/length(idx);
P2=P2/length(idx);


mi=sum(sum(Perf.*log2((Perf+eps)./(P1*P2'+eps))));

H1=-sum(P1.*log2(P1+eps));
H2=-sum(P2.*log2(P2+eps));
mi=mi/(max(H1,H2)+eps);
