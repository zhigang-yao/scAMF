import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score
from transform import transform
from manfit import manfit
from uvi import uvi
from vis import vis

def scAMF(fea_raw, label_1_numr, knn):
    Algs = [
        'SpectralCluster',
        'GAAA',
        'GAAS',
        'GAACe'
    ]

    n_algs = len(Algs)

    Res = np.zeros((4, 3, n_algs))

    fea_trans = transform(fea_raw, 'value2trans')
    fea_trans_fit = manfit(fea_trans, knn)

    fea_cos = transform(fea_raw, 'cosine')
    fea_cos_fit = manfit(fea_cos, knn)

    fea_log = transform(fea_raw, 'log')
    fea_log_fit = manfit(fea_log, knn)

    label_1_numr += 1
    n_class = len(np.unique(label_1_numr))
    ppx = min(50, len(label_1_numr) - 1)

    UVI_highest = 0
    cluster_selected = None
    SI = None

    for i_alg, alg_name in enumerate(Algs):
        print(f'  {alg_name}:')

        if alg_name == 'SpectralCluster':
            # Spectral Clustering
            idx_trans_fit = SpectralClustering(n_clusters=n_class, affinity='nearest_neighbors', random_state=42).fit_predict(fea_trans_fit)
            idx_cos_fit = SpectralClustering(n_clusters=n_class, affinity='nearest_neighbors', random_state=42).fit_predict(fea_cos_fit)
            idx_log_fit = SpectralClustering(n_clusters=n_class, affinity='nearest_neighbors', random_state=42).fit_predict(fea_log_fit)

            # Compute performance metrics
            ACC_trans_fit = accuracy_score(label_1_numr, idx_trans_fit)
            NMI_trans_fit = normalized_mutual_info_score(label_1_numr, idx_trans_fit)
            ARI_trans_fit = adjusted_rand_score(label_1_numr, idx_trans_fit)

            ACC_cos_fit = accuracy_score(label_1_numr, idx_cos_fit)
            NMI_cos_fit = normalized_mutual_info_score(label_1_numr, idx_cos_fit)
            ARI_cos_fit = adjusted_rand_score(label_1_numr, idx_cos_fit)

            ACC_log_fit = accuracy_score(label_1_numr, idx_log_fit)
            NMI_log_fit = normalized_mutual_info_score(label_1_numr, idx_log_fit)
            ARI_log_fit = adjusted_rand_score(label_1_numr, idx_log_fit)

            # Update Res with computed metrics
            Res[:, :, i_alg] = np.array([
                [ACC_trans_fit, NMI_trans_fit, ARI_trans_fit],
                [ACC_cos_fit, NMI_cos_fit, ARI_cos_fit],
                [ACC_log_fit, NMI_log_fit, ARI_log_fit],
            ])

        elif alg_name == 'GAAA':
            # Placeholder for GAAA algorithm
            idx_trans_fit = GAAA(fea_trans_fit, n_class)
            idx_cos_fit = GAAA(fea_cos_fit, n_class)
            idx_log_fit = GAAA(fea_log_fit, n_class)

            # Compute and store performance metrics
            ACC_trans_fit = accuracy_score(label_1_numr, idx_trans_fit)
            NMI_trans_fit = normalized_mutual_info_score(label_1_numr, idx_trans_fit)
            ARI_trans_fit = adjusted_rand_score(label_1_numr, idx_trans_fit)

            ACC_cos_fit = accuracy_score(label_1_numr, idx_cos_fit)
            NMI_cos_fit = normalized_mutual_info_score(label_1_numr, idx_cos_fit)
            ARI_cos_fit = adjusted_rand_score(label_1_numr, idx_cos_fit)

            ACC_log_fit = accuracy_score(label_1_numr, idx_log_fit)
            NMI_log_fit = normalized_mutual_info_score(label_1_numr, idx_log_fit)
            ARI_log_fit = adjusted_rand_score(label_1_numr, idx_log_fit)

            Res[:, :, i_alg] = np.array([
                [ACC_trans_fit, NMI_trans_fit, ARI_trans_fit],
                [ACC_cos_fit, NMI_cos_fit, ARI_cos_fit],
                [ACC_log_fit, NMI_log_fit, ARI_log_fit],
            ])

        elif alg_name == 'GAAS':
            # Placeholder for GAAS algorithm
            idx_trans_fit = GAAS(fea_trans_fit, n_class)
            idx_cos_fit = GAAS(fea_cos_fit, n_class)
            idx_log_fit = GAAS(fea_log_fit, n_class)

            # Compute and store performance metrics
            ACC_trans_fit = accuracy_score(label_1_numr, idx_trans_fit)
            NMI_trans_fit = normalized_mutual_info_score(label_1_numr, idx_trans_fit)
            ARI_trans_fit = adjusted_rand_score(label_1_numr, idx_trans_fit)

            ACC_cos_fit = accuracy_score(label_1_numr, idx_cos_fit)
            NMI_cos_fit = normalized_mutual_info_score(label_1_numr, idx_cos_fit)
            ARI_cos_fit = adjusted_rand_score(label_1_numr, idx_cos_fit)

            ACC_log_fit = accuracy_score(label_1_numr, idx_log_fit)
            NMI_log_fit = normalized_mutual_info_score(label_1_numr, idx_log_fit)
            ARI_log_fit = adjusted_rand_score(label_1_numr, idx_log_fit)

            Res[:, :, i_alg] = np.array([
                [ACC_trans_fit, NMI_trans_fit, ARI_trans_fit],
                [ACC_cos_fit, NMI_cos_fit, ARI_cos_fit],
                [ACC_log_fit, NMI_log_fit, ARI_log_fit],
            ])

        elif alg_name == 'GAACe':
            # Placeholder for GAACe algorithm
            idx_trans_fit = GAACe(fea_trans_fit, n_class)
            idx_cos_fit = GAACe(fea_cos_fit, n_class)
            idx_log_fit = GAACe(fea_log_fit, n_class)

            # Compute and store performance metrics
            ACC_trans_fit = accuracy_score(label_1_numr, idx_trans_fit)
            NMI_trans_fit = normalized_mutual_info_score(label_1_numr, idx_trans_fit)
            ARI_trans_fit = adjusted_rand_score(label_1_numr, idx_trans_fit)

            ACC_cos_fit = accuracy_score(label_1_numr, idx_cos_fit)
            NMI_cos_fit = normalized_mutual_info_score(label_1_numr, idx_cos_fit)
            ARI_cos_fit = adjusted_rand_score(label_1_numr, idx_cos_fit)

            ACC_log_fit = accuracy_score(label_1_numr, idx_log_fit)
            NMI_log_fit = normalized_mutual_info_score(label_1_numr, idx_log_fit)
            ARI_log_fit = adjusted_rand_score(label_1_numr, idx_log_fit)

            Res[:, :, i_alg] = np.array([
                [ACC_trans_fit, NMI_trans_fit, ARI_trans_fit],
                [ACC_cos_fit, NMI_cos_fit, ARI_cos_fit],
                [ACC_log_fit, NMI_log_fit, ARI_log_fit],
            ])

        # Calculate UVI and check if it's the highest
        uvi_trans = uvi(label_1_numr, fea_trans_fit)
        uvi_cos = uvi(label_1_numr, fea_cos_fit)
        uvi_log = uvi(label_1_numr, fea_log_fit)

        if uvi_trans > UVI_highest:
            UVI_highest = uvi_trans
            cluster_selected = idx_trans_fit

        if uvi_cos > UVI_highest:
            UVI_highest = uvi_cos
            cluster_selected = idx_cos_fit

        if uvi_log > UVI_highest:
            UVI_highest = uvi_log
            cluster_selected = idx_log_fit

    # Select the best performing metrics
    ACC_selected = np.max(Res[:, 0, :])
    NMI_selected = np.max(Res[:, 1, :])
    ARI_selected = np.max(Res[:, 2, :])

    # Calculate silhouette score using the best cluster
    SI = vis(fea_trans_fit, cluster_selected, ppx, label_1_numr)

    return Res, ACC_selected, NMI_selected, ARI_selected, cluster_selected, UVI_highest, SI
