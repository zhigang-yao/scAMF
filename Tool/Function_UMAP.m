function Y = Function_UMAP(fea,Y)

c1 = 20; c2 = 20;

n_epochs = 500;

min_dist = 0.1;

n = size(fea,1); 

A = graph_construct(fea',2*floor(log2(n)) + 25);

%fit
D_Y = pdist2(Y,Y);
spread = max(D_Y,[],'all');
phi = fittype('1/(1+a*x^(2*b))','independent','x','coefficients',{'a','b'});
x_fit = linspace(0,spread*3,300);
y_fit = exp((-x_fit+min_dist));
y_fit(x_fit<=min_dist) = 1;
cfun = fit(x_fit',y_fit',phi,'StartPoint',[1 1],'Algorithm', 'Levenberg-Marquardt');
a = cfun.a; b = cfun.b;

%embedding
Y = optimize_layout_euclidean(Y,n_epochs,a,b,A,c1,c2);
Y = (Y-min(Y))./(max(Y)-min(Y));
