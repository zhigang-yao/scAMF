function [ac,map]=calAC(idx,label)

idx = idx +(1-min(idx));
label = label +(1-min(label));

n = max([label(:);idx(:)]);
n = double(n);

Perf = zeros(n);
idx=idx(:); label = label(:);

for i=1:n
    for j=1:n
        Perf(i,j)=sum(abs((idx==i)-(label==j)));
    end
end

[Matching,~] = Hungarian(Perf);
map = Matching*(1:n)';
ac = sum((label)==map(idx))/length(idx);
