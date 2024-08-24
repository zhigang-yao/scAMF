function [embedding] = UMAP_each_point(embedding,A,a,b,alpha,c1,c2)
n = size(embedding,1);
for ii = 1:n
    temp = find(A(ii,:)~=0);
     current = embedding(ii,:);
    for fakeloop = 1:c1
        j = randi(length(temp));
        jj = temp(j);
        if rand()<A(ii,jj)
            other = embedding(jj,:);
            dist_squared = norm(current-other)^2;
            if dist_squared~=0
                grad_coeff = -2*a*b*dist_squared^(b-1);
                grad_coeff = grad_coeff/(a*dist_squared^b+1);
            else
                grad_coeff = 0;
            end
            for d = 1:2
                grad_d = clip(grad_coeff*(current(d)-other(d)));
                current(d) = current(d)+alpha*grad_d;
            end
        end
    end
    for fakeloop = 1:c2
        j = randi(n);
        jj = j;
        if jj == ii
            break
        end
        if rand()<1-A(ii,jj)
            other = embedding(jj,:);
            dist_squared = norm(current-other)^2;
            if dist_squared~=0
                grad_coeff = 2*b;
                grad_coeff = grad_coeff/(0.001+dist_squared)/(a*dist_squared^b+1);
            elseif jj == ii
                continue
            else
                grad_coeff = 0;
            end
            for d = 1:2
                if grad_coeff>0
                    grad_d = clip(grad_coeff*(current(d)-other(d)));
                else
                    grad_d = 4;
                end
                current(d) = current(d)+alpha*grad_d;
            end
        end
    end
embedding(ii,:) = current;
end

end
