import numpy as np

def uvi(labels, S):
    # Calculate the label matrix
    MatLabel = np.equal.outer(labels, labels)

    # Ensure the matrix is double (float in Python)
    MatLabel = MatLabel.astype(float)

    # Calculate uvi according to the formula
    uvi_value = (np.sum(S * MatLabel) / np.sum(MatLabel)) / (np.sum(S * ~MatLabel) / np.sum(~MatLabel) + np.finfo(float).eps)

    return uvi_value