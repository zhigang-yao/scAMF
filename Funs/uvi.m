function uvi = uvi(labels, S)

MatLabel = labels - labels'; 

MatLabel = MatLabel == 0; 

MatLabel = double(MatLabel);

uvi = (sum(sum(S.*MatLabel))/sum(MatLabel(:)))/(sum(sum(S.*~MatLabel))/sum(~MatLabel(:))+eps);

end
