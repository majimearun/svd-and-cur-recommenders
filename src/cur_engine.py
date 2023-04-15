import numpy as np
import pandas as pd


import numpy as np


import numpy as np


def cur_decomposition(matrix, rank):
    """
    Performs CUR decomposition of a given matrix to extract its low-rank approximation.

    Parameters:
        matrix (numpy array): input matrix to be decomposed
        rank (int): desired rank of the approximation

    Returns:
        A tuple (C, U, R) representing the decomposition of the input matrix, where
            - C: a m x rank matrix containing selected columns of the input matrix
            - U: a rank x rank matrix containing the intersection of the selected rows and columns
            - R: a rank x n matrix containing selected rows of the input matrix
    """

    m, n = matrix.shape

    # Check if the rank is greater than the dimensions of the matrix
    if rank > min(m, n):
        rank = min(m, n)

    # Step 1: Select columns based on their importance
    col_frobenius_norms = np.linalg.norm(matrix, axis=0) ** 2
    col_probabilities = col_frobenius_norms / np.sum(col_frobenius_norms)

    selected_col_indices = np.random.choice(
        n, size=rank, p=col_probabilities, replace=False
    )
    selected_col_matrix = matrix[:, selected_col_indices]
    c_scaling_factors = np.sqrt(rank * col_probabilities[selected_col_indices])

    # Step 2: Select rows based on their importance
    row_frobenius_norms = np.linalg.norm(matrix, axis=1) ** 2
    row_probabilities = row_frobenius_norms / np.sum(row_frobenius_norms)

    selected_row_indices = np.random.choice(
        m, size=rank, p=row_probabilities, replace=False
    )
    selected_row_matrix = matrix[selected_row_indices, :]
    r_scaling_factors = np.sqrt(rank * row_probabilities[selected_row_indices])

    # Step 3: Compute the intersection matrix
    intersection_matrix = matrix[
        selected_row_indices[:, np.newaxis], selected_col_indices
    ]
    w, s, vt = np.linalg.svd(intersection_matrix)

    # Compute the pseudo-inverse of the singular values
    pseudo_inv_s = np.diag(1 / s)
    pseudo_inv_s[s < 1e-15] = 0  # Set very small singular values to zero

    # Step 4: Compute the final decomposition
    c = np.dot(selected_col_matrix, np.diag(c_scaling_factors))
    r = np.dot(np.diag(r_scaling_factors), selected_row_matrix)
    u = np.dot(vt.T, np.dot(pseudo_inv_s, np.dot(pseudo_inv_s, w.T)))

    return c, u, r
