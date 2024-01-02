%%Construct a graph of UMAP
function [W,idx] = graph_construct(X,knn)
    %X is the sample matrix with a column is a sample, knn is the number of
    %neighbour by controlling the sparse graph
    n = size(X,2);
    [idx,D_idx] = knnsearch(X',X','K',knn,'Distance','spearman');
    rho = max(D_idx(:,2),1.e-8);
    sigma = zeros(n,1);
    D_n = -max(0,D_idx-rho);
    log2k = log2(knn);
    D_full = -inf*ones(n,n);
    for ii = 1:n
        temp = @(x) sum(exp(D_n(ii,:)/x))-log2k;
        options = optimset('Display','off');
        sol = fsolve(temp,rho(ii),options);
        sigma(ii) = sol;
        D_full(ii,idx(ii,:)) = D_n(ii,:);
    end
    W = exp(bsxfun(@rdivide,D_full,sigma+eps));
    W = W+W'-W.*W';
    W(isnan(W))=0;
end
