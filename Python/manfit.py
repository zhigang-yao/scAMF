import numpy as np
from scipy.spatial.distance import cdist
from transform import transform

def manfit(sample, knn):
    Mout = sample.copy()
    N = sample.shape[0]

    D = cdist(sample, sample, metric='correlation')

    knn3 = 10 * knn
    knn3 = min(knn3, D.shape[1] - 1)
    Nb_dist = np.argsort(D, axis=1)[:, :knn3]
    n = D.shape[0]

    DI = np.zeros((n, n))
    for ii in range(n):
        for jj in Nb_dist[ii]:
            DI[ii, jj] = len(set(Nb_dist[ii, :knn]).intersection(Nb_dist[jj, :knn]))

    DI = (knn - np.maximum(DI, DI.T)) / knn
    D = DI

    Nb_dist = np.argsort(D, axis=1)[:, :knn]

    # Apply the 'value2trans' transformation to the sample
    sample_ = transform(sample, 'value2trans')

    weights = np.array([-0.1, -0.05, 0, 0.05, 0.1])

    for ii in range(N):
        BNbr = sample_[Nb_dist[ii], :]

        xbar = np.mean(BNbr, axis=0)

        d = xbar - sample_[ii, :]

        x_final = xbar

        ds_final = np.sum(cdist(x_final[None, :], BNbr) ** 2)

        for weight in weights:
            x_temp = xbar + weight * d

            ds = np.sum(cdist(x_temp[None, :], BNbr) ** 2)

            if ds <= ds_final:
                x_final = x_temp
                ds_final = ds

        Mout[ii, :] = x_final

    return Mout
