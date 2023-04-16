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


def cur_predict(
    user_ratings: pd.Series,
    movie_id: int,
    U: pd.DataFrame,
    R: pd.DataFrame,
) -> float:
    """
    Function to predict the rating for a given user-movie pair.

    Args:
        user_ratings (numpy array): user ratings
        movie_id (int): movie id
        U (numpy array): U matrix from the CUR decomposition
        R (numpy array): R matrix from the CUR decomposition

    Returns:
        float: predicted rating for the given user-movie pair
    """
    R_m = R[movie_id].values
    u_new = user_ratings @ R.T @ np.linalg.inv(U)
    return u_new.reshape(1, -1) @ R_m.reshape(-1, 1)
