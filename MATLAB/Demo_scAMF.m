warning off;
addpath(genpath('Dataset'));
addpath(genpath('Tool'));
addpath(genpath('Funs'));


knn = 15;

Datasets = {
    1,'Yan', 'GSE36552';
    2,'Goolam', 'EMTAB3321';
    3,'Pollen', 'SRP041736';
    };

n_datasets = size(Datasets,1); 

Res = zeros(n_datasets,5);

for id = 1:3
    data = Datasets{id,3};
    eval(['load ' data]);
    disp(Datasets{id,2});
    rng default;

    [~,ACC_selected,  NMI_selected,  ARI_selected, cluster_selected, UVI_highest,SI] = scAMF(fea_raw,label_1_numr,knn);

    if UVI_highest<10
        fea_raw = GeneFilter(fea_raw, 5,'correlation');
        disp('The highest UVI is less than 10, Do gene filter and rerun scAMF.')
        [~,ACC_selected,  NMI_selected,  ARI_selected, cluster_selected, UVI_highest,SI] = scAMF(fea_raw,label_1_numr,knn);
    end

    Res(id,:) = [ACC_selected,  NMI_selected,  ARI_selected, UVI_highest,SI];
end
