import numpy as np
import pandas as pd


def cur_decomposition(
    matrix: pd.DataFrame, rank: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Computes the CUR decomposition of a matrix.

    Args:
        matrix (pd.DataFrame): matrix to decompose
        rank (int): rank of the decomposition

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: C, U, R matrices
    """
    m, n = matrix.shape
    ratings = matrix
    matrix = matrix.values
    # Select columns for C matrix
    col_norms = np.linalg.norm(matrix, axis=0)
    col_probs = col_norms / np.sum(col_norms)
    col_indices = np.random.choice(np.arange(n), rank, replace=False, p=col_probs)
    C = matrix[:, col_indices] / np.sqrt(rank * col_probs[col_indices])

    # Select rows for R matrix
    row_norms = np.linalg.norm(matrix, axis=1)
    row_probs = row_norms / np.sum(row_norms)
    row_indices = np.random.choice(np.arange(m), rank, replace=False, p=row_probs)
    R = matrix[row_indices, :] / np.sqrt(rank * row_probs[row_indices][:, np.newaxis])

    # Compute U matrix as the pseudoinverse of C
    U = np.linalg.pinv(C) @ matrix @ np.linalg.pinv(R)

    C = pd.DataFrame(C, index=ratings.index)
    R = pd.DataFrame(R, columns=ratings.columns)
    U = pd.DataFrame(U)

    return C, U, R
