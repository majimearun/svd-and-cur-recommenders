import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from svd_engine import rmse, top_k_precision


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

    if rank > min(m, n):
        rank = min(m, n)
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


def non_zero_rmse(
    matrix: pd.DataFrame, C: pd.DataFrame, U: pd.DataFrame, R: pd.DataFrame
) -> float:
    """
    Calculates the RMSE of the non-zero elements of a matrix.

    Args:
        matrix (pd.DataFrame): matrix to calculate the RMSE
        C (pd.DataFrame): C matrix of the CUR decomposition
        U (pd.DataFrame): U matrix of the CUR decomposition
        R (pd.DataFrame): R matrix of the CUR decomposition

    Returns:
        float: RMSE of the non-zero elements of the matrix
    """
    A_pred = C @ U @ R
    A_pred = A_pred.apply(lambda x: np.clip(x, 0, 5), axis=1)
    A_pred = A_pred.apply(lambda x: np.round(x), axis=1)
    A_pred = A_pred.values
    non_zero = matrix.nonzero()
    non_zero_pred = A_pred[non_zero]
    matrix_non_zero = matrix[non_zero]
    return np.sqrt(np.mean((non_zero_pred - matrix_non_zero) ** 2))


def cur_predict(
    user_id: int, movie_id: int, C: pd.DataFrame, U: pd.DataFrame, R: pd.DataFrame
) -> float:
    """
    Predicts the rating of a user for a movie using the CUR decomposition.

    Args:
        user_id (int): user id
        movie_id (int): movie id
        C (pd.DataFrame): C matrix of the CUR decomposition
        U (pd.DataFrame): U matrix of the CUR decomposition
        R (pd.DataFrame): R matrix of the CUR decomposition

    Returns:
        float: predicted rating
    """
    return np.clip((C.loc[user_id] @ U @ R[movie_id]).item(), 0, 5)


def cur_score(
    test: pd.DataFrame,
    C: pd.DataFrame,
    U: pd.DataFrame,
    R: pd.DataFrame,
    k: int = 5,
) -> None:
    """
    Scores the RMSE and top-k precision for the given test set.

    Args:
        test (pd.DataFrame): test set
        C (pd.DataFrame): C matrix of the CUR decomposition
        U (pd.DataFrame): U matrix of the CUR decomposition
        R (pd.DataFrame): R matrix of the CUR decomposition
        k (int): number of top items to consider. Defaults to 5.
    Returns:
        None

    """
    actual = []
    actual_per_user = []
    preds_per_user = []
    avg_top_k_precision = 0
    preds = []
    for user_id, movie in test.iterrows():
        for movie_id, rating in movie.items():
            if rating != 0:
                pred = cur_predict(user_id, movie_id, C, U, R)
                actual.append((movie_id, rating))
                actual_per_user.append((movie_id, rating))
                preds.append((movie_id, pred))
                preds_per_user.append((movie_id, pred))
        avg_top_k_precision += top_k_precision(actual_per_user, preds_per_user, k)
        actual_per_user = []
        preds_per_user = []
    actual = np.array(actual)
    preds = np.array(preds)
    print(actual.shape, preds.shape)
    print(f"RMSE: {rmse(actual[:, 1], preds[:, 1])}")
    print(f"Top {k} precision: {(avg_top_k_precision / len(test))*100}%")
    print(f"Number of latent factors: {len(U)}")
    print(f"Spearmans correlation: {spearmanr(actual[:, 1], preds[:, 1])[0]}")
