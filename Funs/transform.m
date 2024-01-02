function Matrix_T = transform(Matrix,type)

switch type
    case 'value2trans'
        [~, sortedIndices] = sort(Matrix, 2);

        OrdinalNumbers = repmat(1:size(Matrix, 2), size(Matrix, 1), 1);

        linearIndex = sub2ind(size(Matrix), repmat((1:size(Matrix, 1))', 1, size(Matrix, 2)), sortedIndices);

        Matrix_T = zeros(size(Matrix)); Matrix_T(linearIndex) = (OrdinalNumbers/size(Matrix, 2)).^0.5;

    case 'cosine'
        Matrix_T = Matrix./sqrt(sum(Matrix.^2,2));

    case 'log'
        Matrix_T = log2(Matrix + 1);

end
