import numpy as np

def transform(Matrix, type_):
    if type_ == 'value2trans':
        sorted_indices = np.argsort(Matrix, axis=1, kind='stable')
        ordinal_numbers = np.tile(np.arange(1, Matrix.shape[1] + 1), (Matrix.shape[0], 1))
        linear_index = np.arange(Matrix.shape[0])[:, None] * Matrix.shape[1] + sorted_indices
        Matrix_T = np.zeros_like(Matrix)
        Matrix_T.flat[linear_index] = (ordinal_numbers / Matrix.shape[1]) ** 0.5

    elif type_ == 'cosine':
        Matrix_T = Matrix / np.sqrt(np.sum(Matrix ** 2, axis=1, keepdims=True))

    elif type_ == 'log':
        Matrix_T = np.log2(Matrix + 1)

    return Matrix_T