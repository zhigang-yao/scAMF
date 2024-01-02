# Single-Cell Analysis via Manifold Fitting: A New Framework for RNA Clustering and Beyond

## Pre-requisites
- MATLAB R2022b

## How to use scAMF
- Load the data set: 
  - `fea_raw`: the raw expression matrix; 
  - `label_1_numr`: the labels;
- Generate initial scAMF results: `[Res, ACC_selected, NMI_selected, ARI_selected, cluster_selected, UVI_highest, SI] = scAMF(fea_raw, label_1_numr, knn)`
  - `Res`: the results that contain ACC, NMI, ARI, and UVI values for each combination of transformation methods and clustering algorithms;
  - `cluster_selected`: the final clustering results selected out of all combinations;
  - `UVI_highest`: the corresponding UVI to cluster_selected;
  - `ACC_selected`, `NMI_selected`, `ARI_selected`, `UVI_highest`, `SI`: ACC, NMI, ARI, UVI, and SI calculated based on cluster_selected;
- Test if `UVI_highest` is smaller than 10: 
  - If so, rerun scAMF and generate the final results;
  - If not, output the initial results as the final ones;

## Citation