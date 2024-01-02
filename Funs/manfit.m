function Mout = manfit(sample, knn)

Mout  = sample;  N = size(sample,1); 

D = pdist2(sample,sample,"correlation");

knn3 = 10*knn; [~, Nb_dist] = mink(D,knn3);  n = size(D,1); 

DI = zeros(n);
for ii = 1:n
    for jj = Nb_dist(:,ii)'
       DI(ii,jj) = length(intersect(Nb_dist(1:knn,ii), Nb_dist(1:knn,jj))); 
    end
end
DI = (knn - max(DI,DI'))/knn;
D = DI;

[~, Nb_dist] = mink(D,knn); 

sample_ = single(transform(sample,'value2trans'));

parfor ii = 1:N
    
    BNbr = sample_(Nb_dist(:,ii),:);

    xbar = mean(BNbr,1);

    d = xbar - sample_(ii,:);

    weights = [-0.1,-0.05,0,0.05,0.1]; 

    x_final = xbar;

    ds_final = sum(pdist2(x_final,sample_(Nb_dist(:,ii),:)).^2);

    for pp = 1:5

        x_temp = xbar + weights(pp)*d;

        ds = sum(pdist2(x_temp,sample_(Nb_dist(:,ii),:)).^2);

        if ds <= ds_final

            x_final = x_temp;

            ds_final = ds;

        end
    
    end

    Mout(ii,:) = x_final;

end




