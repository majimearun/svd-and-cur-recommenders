import numpy as np
import pandas as pd
from scipy.linalg import svd
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr


def read_data(filename: str = "../data/u.data") -> pd.DataFrame:
    """
    Reads the movielens100k dataset from a .data file.

    Args:
        filename (str): path to the file to read. Defaults to "../data/u.data".

    Returns:
        pandas.DataFrame: a dataframe containing the data (columns: user_id, movie_id, rating, timestamp)
    """
    user_ids = []
    movie_ids = []
    ratings = []
    timestamps = []

    with open(filename, "rt") as file1:
        for line in file1.readlines():
            a = line.split()
            user_ids.append(int(a[0]))
            movie_ids.append(int(a[1]))
            ratings.append(float(a[2]))
            timestamps.append(a[3])

    rating_df = pd.DataFrame(
        {
            "user_id": user_ids,
            "movie_id": movie_ids,
            "rating": ratings,
            "timestamp": timestamps,
        }
    )
    rating_df.sort_values(by=["user_id", "movie_id", "timestamp"], inplace=True)
    rating_df.reset_index(drop=True, inplace=True)
    return rating_df


def keep_movies_rated_by_at_least(df: pd.DataFrame, perc: float = 0.0) -> pd.DataFrame:
    """
    Filters the dataframe to keep only movies that have been rated by at least perc% of the users.

    Args:
        df (pandas.DataFrame): a dataframe containing the data (columns: user_id, movie_id, rating, timestamp)
        perc (float): percentage of users that have rated a movie. Defaults to 0.0. (0 <= perc <= 1)

    Returns:
        pandas.DataFrame: a dataframe containing the data (columns: user_id, movie_id, rating, timestamp)
    """
    filtered = df.groupby("movie_id").filter(
        lambda x: len(x) >= perc * df.user_id.nunique()
    )
    return filtered


def create_pivot_table(data: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a pivot table from the dataframe, where rowsa re the users and columns are the movies. Values are the ratings corresponding to the user-movie pair. (ratings for watched movies lie between 1 and 5, unwatched are filled with 0).

    Args:
        data (pandas.DataFrame): a dataframe containing the data (columns: user_id, movie_id, rating, timestamp)

    Returns:
        pandas.DataFrame: a dataframe containing the data (rows: user_id, columns: movie_id, values: rating)
    """
    temp = data.pivot_table(
        index="user_id", columns="movie_id", values="rating"
    ).fillna(0)
    return pd.DataFrame(
        columns=temp.columns.values, index=temp.index.values, data=temp.values
    )


def split(data: pd.DataFrame, split: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function to split the data into train and test sets.

    Args:
        data (pandas.DataFrame): a dataframe containing the data (rows: user_id, columns: movie_id, values: rating)
        split (float): percentage of data to be used for testing. Defaults to 0.2. (0 <= split <= 1)

    Returns:
        tuple(pandas.DataFrame, pandas.DataFrame): a tuple containing the train and test sets
    """
    train, test = train_test_split(data, test_size=split, random_state=42)
    return train, test


def preserve_variance(
    perc: float,
    U: np.ndarray,
    VT: np.ndarray,
    sigma: np.ndarray,
    users: pd.Series.index,
    movies: pd.DataFrame.columns,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """
    A utility function to preserve a certain percentage of the variance in the SVD decomposition.

    Args:
        perc (float): percentage of variance to preserve. Defaults to 0.0. (0 <= perc <= 1)
        U (numpy.ndarray): U matrix from the SVD decomposition (user X latent factors)
        VT (numpy.ndarray): VT matrix from the SVD decomposition (latent factors X movie)
        sigma (numpy.ndarray): Diagnol sigma matrix (strength of latent factors) from the SVD decomposition (latent factors X latent factors)
        users (pd.Series.index): index of the users in the pivot table, to be used as the index for the U matrix
        movies (pd.DataFrame.columns): index of the movies in the pivot table, to be used as the columns for the VT matrix

    Returns:
        tuple(pd.DataFrame, pd.DataFrame, pd.DataFrame, int): a tuple containing the sigma, U and VT matrices after preserving the variance and the number of latent factors

    """
    rows_to_remove = 0
    preserved = np.sum(sigma)
    preserved_copy = preserved
    for i in range(sigma.shape[1] - 1, -1, -1):
        preserved -= sigma[i, i]
        if preserved / preserved_copy < perc:
            rows_to_remove = i + 1
            break
        elif preserved / preserved_copy == perc:
            rows_to_remove = i
            break
    k = rows_to_remove
    return (
        pd.DataFrame(sigma[:k, :k]),
        pd.DataFrame(U[:, :k], index=users),
        pd.DataFrame(VT[:k, :], columns=movies),
        k,
    )


def compute_svd(
    data: pd.DataFrame, perc: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Function to compute the SVD decomposition of the pivot table.

    Args:
        data (pandas.DataFrame): a dataframe containing the data (rows: user_id, columns: movie_id, values: rating)
        perc (float): percentage of variance to preserve. Defaults to 0.0. (0 <= perc <= 1)

    Returns:
        tuple(pd.DataFrame, pd.DataFrame, pd.DataFrame, int): a tuple containing the U, sigma and VT matrices after preserving the variance and the number of latent factors
    """
    U, Sigma, VT = svd(data)
    Sigma = np.diag(Sigma)
    users = data.index
    movies = data.columns
    sigma, U, VT, k = preserve_variance(perc, U, VT, Sigma, users, movies)
    return U, sigma, VT, k


def rmse(true: np.ndarray, pred: np.ndarray) -> float:
    """
    Function to compute the RMSE between the true and predicted values.

    Args:
        true (numpy.ndarray): true values
        pred (numpy.ndarray): predicted values

    Returns:
        float: RMSE value for the given true and predicted values
    """
    x = true - pred
    return np.sqrt(np.mean(x**2))


def cosine_similarity(y1: np.ndarray, y2: np.ndarray) -> float:
    """
    Function to compute the cosine similarity between two vectors.

    Args:
        y1 (numpy.ndarray): first vector
        y2 (numpy.ndarray): second vector

    Returns:
        float: cosine similarity between the two vectors
    """
    return np.dot(y1, y2) / (np.linalg.norm(y1) * np.linalg.norm(y2))


def clean_preds(y: np.ndarray) -> np.ndarray:
    """
    A function to clean the predicted values as given by the SVD decomposition algorithm. We do not want negative ratings or ratings greater than 5. Therefore, we set the negative ratings to 0 and the ratings greater than 5 to 5.

    Args:
        y (numpy.ndarray): predicted values

    Returns:
        numpy.ndarray: cleaned predicted values
    """
    y_final = []
    for x in y:
        if x[1] > 0:
            if x[1] > 5:
                y_final.append((x[0], 5))
            else:
                y_final.append(x)
        else:
            y_final.append((x[0], 0))
    return y_final


def svr_predict(
    user_id: int,
    user_ratings: np.ndarray,
    movie_id: int,
    U: pd.DataFrame,
    sigma: pd.DataFrame,
    VT: pd.DataFrame,
) -> float:
    """
    Function to predict the rating for a given user-movie pair.

    Args:
        user_id (int): user id
        movie_id (int): movie id
        U (pd.DataFrame): U matrix from the SVD decomposition (user X latent factors)
        sigma (pd.DataFrame): Diagnol sigma matrix (strength of latent factors) from the SVD decomposition (latent factors X latent factors)
        VT (pd.DataFrame): VT matrix from the SVD decomposition (latent factors X movie)

    Returns:
        float: predicted rating for the given user-movie pair
    """
    V = VT[movie_id].values
    if user_id not in U.index:
        u_new = user_ratings @ VT.T @ np.linalg.inv(sigma)
    else:
        u_new = U[user_id].value
    return u_new.reshape(1, -1) @ sigma.values @ V.reshape(-1, 1)


def top_k_precision(actual: list, predicted: list, k: int) -> float:
    """
    Function to compute the top-k precision for a given user.

    Args:
        actual (list): list of tuples containing the actual ratings for the user
        predicted (list): list of tuples containing the predicted ratings for the user
        k (int): number of top items to consider

    Returns:
        float: top-k precision for the given user
    """
    actual = sorted(actual, key=lambda x: x[1], reverse=True)
    predicted = sorted(predicted, key=lambda x: x[1], reverse=True)
    actual = [x[0] for x in actual[:k]]
    predicted = [x[0] for x in predicted[:k]]
    return len(set(actual) & set(predicted)) / k


def svd_score(
    test: pd.DataFrame,
    U: pd.DataFrame,
    sigma: pd.DataFrame,
    VT: pd.DataFrame,
    k: int = 5,
) -> None:
    """
    Scores the RMSE and top-k precision for the given test set.

    Args:
        test (pd.DataFrame): test set
        U (pd.DataFrame): U matrix from the SVD decomposition (user X latent factors)
        sigma (pd.DataFrame): Diagnol sigma matrix (strength of latent factors) from the SVD decomposition (latent factors X latent factors)
        VT: (pd.DataFrame): VT matrix from the SVD decomposition (latent factors X movie)
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
                pred = svr_predict(
                    user_id, test.loc[user_id].values, movie_id, U, sigma, VT
                )
                actual.append((movie_id, rating))
                actual_per_user.append((movie_id, rating))
                preds.append((movie_id, pred))
                preds_per_user.append((movie_id, pred))
        avg_top_k_precision += top_k_precision(actual_per_user, preds_per_user, k)
        actual_per_user = []
        preds_per_user = []
    actual = np.array(actual)
    preds = np.array(clean_preds(preds), dtype="object")
    print(preds)
    print(actual.shape, preds.shape)
    print(f"RMSE: {rmse(actual[:, 1], preds[:, 1])}")
    print(f"Top {k} precision: {(avg_top_k_precision / len(test))*100}%")
    print(f"Number of latent factors: {len(sigma)}")
    print(f"Spearmans correlation: {spearmanr(actual[:, 1], preds[:, 1])[0]}")
