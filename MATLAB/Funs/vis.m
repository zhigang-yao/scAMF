function SI = vis(fea_selected, cluster_selected, ppx, label_1_numr)

num_features = size(fea_selected, 2);
p_values = zeros(num_features, 1);

for i = 1:num_features
    [p, ~, ~] = anova1(fea_selected(:, i), cluster_selected, 'off');
    p_values(i) = p;
end

[~,significant_features] = mink(p_values,min(100*round(log2(size(fea_selected,1))),size(fea_selected,2)));

fea_selected = fea_selected(:,significant_features);

figure; hold on;
Y = tsne(fea_selected,'Perplexity',ppx,'Distance','spearman');
Y = Function_UMAP(fea_selected,Y); gscatter(Y(:,1),Y(:,2),label_1_numr);
si = silhouette(Y,label_1_numr); SI = mean(si);
title(['SI = ', num2str(round(mean(si),2))]);
xlabel('scAMF 1');
ylabel('scAMF 2');
box off;legend off;
ax = gca;
ax.LineWidth = 1.5;

end
