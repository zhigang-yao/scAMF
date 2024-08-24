function [Res,ACC_selected,  NMI_selected, ARI_selected, cluster_selected,UVI_highest, SI] = scAMF(fea_raw,label_1_numr,knn)


Algs = {
    1,'SpectralCluster';...
    2,'GAAA';...
    3,'GAAS';...
    4,'GAACe';...
    };

n_algs = length(Algs);

Res = zeros(4,3,n_algs);

fea_raw = full(fea_raw);
fea_trans = transform(fea_raw,'value2trans');
fea_trans_fit = manfit(fea_trans, knn);
fea_trans_fit = double(fea_trans_fit);

fea_cos  =  transform(fea_raw,'cosine');
fea_cos_fit = manfit(fea_cos, knn);
fea_cos_fit = double(fea_cos_fit);

fea_log =  transform(fea_raw,'log');
fea_log_fit = manfit(fea_log, knn);
fea_log_fit = double(fea_log_fit);

label_1_numr = label_1_numr + 1;
n_class = length(unique(label_1_numr));
ppx = min(50,length(label_1_numr)-1);


UVI_highest = 0;

for i_alg = 1:n_algs
    fprintf('  %2s:\n', Algs{i_alg,2});

    switch Algs{i_alg,2}

        case 'SpectralCluster'

            idx_trans_fit  = spectralcluster(fea_trans_fit,n_class);
            idx_cos_fit    = spectralcluster(fea_cos_fit,n_class);
            idx_log_fit    = spectralcluster(fea_log_fit,n_class);

        case 'GAAA'
            Z = linkage(fea_trans_fit, 'average');
            idx_trans_fit = cluster(Z,'MaxClust',n_class);

            Z = linkage(fea_cos_fit, 'average');
            idx_cos_fit = cluster(Z,'MaxClust',n_class);

            Z = linkage(fea_log_fit, 'average');
            idx_log_fit = cluster(Z,'MaxClust',n_class);

        case 'GAAS'
            Z = linkage(fea_trans_fit, 'single');
            idx_trans_fit = cluster(Z,'MaxClust',n_class);

            Z = linkage(fea_cos_fit, 'single');
            idx_cos_fit = cluster(Z,'MaxClust',n_class);

            Z = linkage(fea_log_fit, 'single');
            idx_log_fit = cluster(Z,'MaxClust',n_class);

        case 'GAACe'
            Z = linkage(fea_trans_fit, 'centroid');
            idx_trans_fit = cluster(Z,'MaxClust',n_class);

            Z = linkage(fea_cos_fit, 'centroid');
            idx_cos_fit = cluster(Z,'MaxClust',n_class);

            Z = linkage(fea_log_fit, 'centroid');
            idx_log_fit = cluster(Z,'MaxClust',n_class);

    end

    idx = idx_trans_fit;
    ACC = calAC(idx,label_1_numr);
    NMI = calMI(idx,label_1_numr);
    ARI = calARI(idx,label_1_numr);

    n = size(fea_raw,1);
    kpp = min(ceil(n/10),40);

    D = pdist2(fea_log,fea_log);
    n = size(D,1); DO = n*ones(n,n);
    nkpps = 1:kpp;
    for ii = 1:n
        [~,a] = mink(D(ii,:),kpp);
        DO(ii,a) = nkpps-1;
    end
    DO = min(DO,DO');
    D = DO;
    D_min = mink(D,kpp,2);
    sigma_q = D_min(:,kpp);
    A = D<=min(sigma_q,sigma_q');
    UVI = uvi(idx, A);
    Res(:, 1, i_alg) = [ACC,NMI,ARI,UVI];

    if UVI>UVI_highest
        cluster_selected = idx;
        UVI_highest = UVI;
    end


    fprintf('Trans           ACC: %2.4f, NMI: %2.4f, ARI: %2.4f, UVI: %2.4f.\n',ACC,NMI,ARI,UVI);

    idx = idx_cos_fit;
    ACC = calAC(idx,label_1_numr);
    NMI = calMI(idx,label_1_numr);
    ARI = calARI(idx,label_1_numr);
    UVI = uvi(idx, A);
    Res(:, 2, i_alg) = [ACC,NMI,ARI,UVI];

    if UVI>UVI_highest
        cluster_selected = idx;
        UVI_highest = UVI;
    end

    fprintf('Cos             ACC: %2.4f, NMI: %2.4f, ARI: %2.4f, UVI: %2.4f.\n',ACC,NMI,ARI,UVI);

    idx = idx_log_fit;
    ACC = calAC(idx,label_1_numr);
    NMI = calMI(idx,label_1_numr);
    ARI = calARI(idx,label_1_numr);
    UVI = uvi(idx, A);
    Res(:, 3, i_alg) = [ACC,NMI,ARI,UVI];

    if UVI>UVI_highest
        cluster_selected = idx;
        UVI_highest = UVI;
    end

    fprintf('Log             ACC: %2.4f, NMI: %2.4f, ARI: %2.4f, UVI: %2.4f.\n',ACC,NMI,ARI,UVI);

end

ACCs = squeeze(Res(1,:,:)); ACCs = ACCs(:);
NMIs = squeeze(Res(2,:,:)); NMIs = NMIs(:);
ARIs = squeeze(Res(3,:,:)); ARIs = ARIs(:);
UVIs = squeeze(Res(4,:,:)); UVIs = UVIs(:);

[UVI_selected, UVI_id] = max(UVIs);
ACC_selected = ACCs(UVI_id);
NMI_selected = NMIs(UVI_id);
ARI_selected = ARIs(UVI_id);

fprintf('\nSelected:       ACC: %2.4f, NMI: %2.4f, ARI: %2.4f, UVI: %2.4f.\n',ACC_selected,NMI_selected,ARI_selected,UVI_selected);

SI = 0;

if UVI_highest>10

    [~, position] = max(max(squeeze(Res(4,:,:)),[],2)); position = position(1);

    if position == 1
        fea_selected = fea_trans;
    elseif position == 2
        fea_selected = fea_cos;
    else
        fea_selected = fea_log;
    end

    SI = vis(fea_selected,cluster_selected, ppx, label_1_numr);

end
