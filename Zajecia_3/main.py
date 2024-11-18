import numpy as np
from sklearn.cluster import KMeans
import json


def create_user_vectors(data):
    """
    Create user vectors representing their ratings for all movies.

    :param data: Dictionary with user ratings. Keys are user IDs, values are dictionaries of movie ratings.
    :return: Tuple of user vectors as a numpy array and a list of all unique movies.
    """
    all_movies = list(set(movie for user in data.values() for movie in user))
    users = list(data.keys())

    user_vectors = np.array([
        [data[user].get(movie, 0) for movie in all_movies]
        for user in users
    ])

    return user_vectors, all_movies, users


def fit_kmeans(user_vectors, n_clusters):
    """
    Fit a KMeans clustering model to the user vectors.

    :param user_vectors: Numpy array of user vectors.
    :param n_clusters: Number of clusters for KMeans.
    :return: Trained KMeans model and cluster labels.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(user_vectors)
    return kmeans, kmeans.labels_


def get_unseen_movies(target_user_ratings, all_movies):
    """
    Identify movies that the target user has not seen.

    :param target_user_ratings: Ratings of the target user as a vector.
    :param all_movies: List of all unique movies.
    :return: List of movies that the target user has not seen.
    """
    return [
        all_movies[i]
        for i, rating in enumerate(target_user_ratings)
        if rating == 0
    ]


def calculate_movie_scores(cluster_users, unseen_movies, user_vectors, all_movies):
    """
    Calculate scores for unseen movies based on cluster users' ratings.

    :param cluster_users: List of indices of users in the target cluster.
    :param unseen_movies: List of movies not seen by the target user.
    :param user_vectors: Numpy array of user vectors.
    :param all_movies: List of all unique movies.
    :return: Dictionary with average scores for each unseen movie.
    """
    movie_scores = {movie: [] for movie in unseen_movies}

    for user_idx in cluster_users:
        for i, movie in enumerate(all_movies):
            if movie in unseen_movies:
                movie_scores[movie].append(user_vectors[user_idx][i])

    return {
        movie: np.mean(scores) if scores else 0
        for movie, scores in movie_scores.items()
    }


def recommend_and_anti_recommend(data, target_user_id, n_clusters=3, n_recommend=5, n_anti=5):
    """
    Generate movie recommendations and anti-recommendations for a target user based on clustering.

    :param data: Dictionary with user ratings. Keys are user IDs, values are dictionaries of movie ratings.
    :param target_user_id: ID of the user to generate recommendations for.
    :param n_clusters: Number of clusters for KMeans.
    :param n_recommend: Number of recommendations to return.
    :param n_anti: Number of anti-recommendations to return.
    :return: Tuple of lists: (recommendations, anti-recommendations).
    """
    user_vectors, all_movies, users = create_user_vectors(data)
    kmeans, clusters = fit_kmeans(user_vectors, n_clusters)

    target_user_idx = users.index(target_user_id)
    target_cluster = clusters[target_user_idx]
    target_user_ratings = user_vectors[target_user_idx]

    unseen_movies = get_unseen_movies(target_user_ratings, all_movies)
    cluster_users = [
        idx for idx, cluster in enumerate(clusters)
        if cluster == target_cluster and idx != target_user_idx
    ]

    avg_movie_scores = calculate_movie_scores(cluster_users, unseen_movies, user_vectors, all_movies)
    sorted_movies = sorted(avg_movie_scores.items(), key=lambda x: x[1], reverse=True)

    recommendations = [movie for movie, score in sorted_movies[:n_recommend]]
    anti_recommendations = [movie for movie, score in sorted_movies[-n_anti:]]

    return recommendations, anti_recommendations


with open('Resources/movies_and_tv_shows.json', 'r') as file:
    data = json.load(file)

recs, anti_recs = recommend_and_anti_recommend(data, "user0")
print("Recommendations for user0:", recs)
print("Anti-recommendations for user0:", anti_recs)
