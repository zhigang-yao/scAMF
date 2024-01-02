function fea_filtered = GeneFilter(fea_raw, knn, dist_type)

n = size(fea_raw,1);

D = pdist2(fea_raw, fea_raw, dist_type);

[~,ID_nbr] = mink(D,knn,2);

G = zeros(n);

for ii = 1:n
    G(ii,ID_nbr(ii,:)) = 1;
end

G = min(G,G');

bins = conncomp(graph(G));

[groups, ~, idx] = unique(bins);

counts = accumarray(idx, 1);

unique_numbers = groups(counts<4);

for ii = unique_numbers
    bins(bins==ii) = 0;
end

fea_grouped = fea_raw(bins>0,:);
cluster_labels = bins(bins>0);

num_features = size(fea_grouped, 2);
p_values = zeros(num_features, 1);

for i = 1:num_features
    [p, ~, ~] = anova1(fea_grouped(:, i), cluster_labels, 'off');
    p_values(i) = p;
end

% % 找出显著的特征（例如，使用0.05作为显著性水平）
[~,significant_features] = mink(p_values,500);

fea_filtered = fea_raw(:,significant_features);
