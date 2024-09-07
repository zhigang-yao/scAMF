import numpy as np
from scipy.stats import f_oneway
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import umap
import matplotlib.pyplot as plt
import seaborn as sns

def vis(fea_selected, cluster_selected, ppx, label_1_numr):
    num_features = fea_selected.shape[1]
    p_values = np.zeros(num_features)

    # ANOVA test for each feature
    for i in range(num_features):
        unique_clusters = np.unique(cluster_selected)
        samples = [fea_selected[cluster_selected == cluster, i] for cluster in unique_clusters]
        p_values[i] = f_oneway(*samples).pvalue

    # Select significant features
    significant_features = np.argsort(p_values)[:min(100 * round(np.log2(fea_selected.shape[0])), fea_selected.shape[1])]

    fea_selected = fea_selected[:, significant_features]

    # Apply t-SNE
    tsne_model = TSNE(perplexity=ppx, metric='spearman', random_state=42)
    Y = tsne_model.fit_transform(fea_selected)

    # Apply UMAP (assuming Function_UMAP is UMAP, adjust if different)
    umap_model = umap.UMAP()
    Y = umap_model.fit_transform(Y)

    # Plot the result
    plt.figure()
    sns.scatterplot(x=Y[:, 0], y=Y[:, 1], hue=label_1_numr, palette="deep")
    plt.title(f'SI = {round(silhouette_score(Y, label_1_numr), 2)}')
    plt.xlabel('scAMF 1')
    plt.ylabel('scAMF 2')
    plt.box(False)
    plt.legend().remove()
    plt.gca().spines['top'].set_linewidth(1.5)
    plt.gca().spines['right'].set_linewidth(1.5)
    plt.show()

    # Calculate silhouette score
    SI = silhouette_score(Y, label_1_numr)

    return SI