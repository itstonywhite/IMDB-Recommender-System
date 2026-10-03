import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import joblib

# Loading the datasets
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# print(f'\nMovies Dataset Shape: {movies.shape}')
# print(f'Ratings Data Shape: {ratings.shape}\n')
# display(movies.head(3))


# Merging datasets on movieId
df = pd.merge(ratings, movies, on='movieId')

# Dropping 'timestamp' as it's not needed for basic Collaborative Filtering
df.drop('timestamp', axis=1, inplace=True)

# print()

# Visualizing top 10 most rated movies
plt.figure(figsize=(10, 5))
top_movies = df.groupby('title')['rating'].count().sort_values(ascending=False).head(10)
bars = sns.barplot(x=top_movies.values, y=top_movies.index, hue=top_movies.index, palette='crest', legend=False)

plt.title('Top 10 Most Rated Movies', fontweight='bold')
plt.xlabel('Number of Ratings', color='lightgray')
plt.ylabel('')
plt.show() # Renders the plot

# Creating a Pivot Table: Rows = Users, Columns = Movie Titles, Values = Ratings
# Using pivot_table with mean aggregation handles rare duplicate titles
user_movie_pivot = df.pivot_table(index='userId', columns='title', values='rating').fillna(0)
movie_user_pivot = user_movie_pivot.T

# Converting to Scipy Sparse Matrix for memory efficiency
user_movie_sparse = csr_matrix(user_movie_pivot.values)
movie_user_sparse = csr_matrix(movie_user_pivot.values)

# print(f'\nUser-Movie Matrix Shape: {user_movie_pivot.shape}\n')


# Cleaning genres: "Action|Comedy" -> "Action Comedy"
movies['genres_clean'] = movies['genres'].str.replace('|', ' ')

# Applying TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres_clean'])

# Computing Cosine Similarity between all movies
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Helper structure for reverse mapping
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def get_content_based_recommendations(title, cosine_sim=cosine_sim):
    """Recommends movies similar to the given movie based on genres (similarities)."""
    if title not in indices:
        return ["Movie not found in the database :("]

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort by highest similarity
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get top 5 similar movies (skipping index 0 which is the movie itself)
    sim_scores = sim_scores[1:6]

    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

# Test the Model
print("\nContent-Based Recommendations for 'Toy Story (1995)':")
for i, movie in enumerate(get_content_based_recommendations('Toy Story (1995)'), 1):
    print(f'{i}. {movie}')

print()


# Initialize KNN models with Cosine distance metric
user_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
item_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)

# Fit models on the sparse matrices
user_knn.fit(user_movie_sparse)
item_knn.fit(movie_user_sparse)

# Item-Based recommendation function
def get_item_based_recommendations(movie_title, pivot=movie_user_pivot, model=item_knn, n_recs=5):
    """Recommends movies similar to the given movie based on user ratings."""
    if movie_title not in pivot.index:
        return ["Movie not found."]

    query_idx = pivot.index.get_loc(movie_title)
    distances, indices = model.kneighbors(pivot.iloc[query_idx, :].values.reshape(1, -1), n_neighbors=n_recs+1)

    recommendations = [pivot.index[indices.flatten()[i]] for i in range(1, len(distances.flatten()))]
    return recommendations

# User-Based recommendation function
def get_user_based_recommendations(target_user_id, pivot=user_movie_pivot, model=user_knn, n_recs=5):
    """Recommends movies to a user based on what similar users enjoyed."""
    if target_user_id not in pivot.index:
        return ["User not found."]

    # Finding similar users
    query_idx = pivot.index.get_loc(target_user_id)
    distances, indices = model.kneighbors(pivot.iloc[query_idx, :].values.reshape(1, -1), n_neighbors=5)
    similar_users = [pivot.index[indices.flatten()[i]] for i in range(1, len(distances.flatten()))]

    # Get movies the target user has already seen
    user_seen_movies = set(pivot.columns[pivot.loc[target_user_id] > 0])

    # Accumulate recommendations from similar users
    rec_scores = {}
    for sim_user in similar_users:
        sim_user_ratings = pivot.loc[sim_user]
        # Looking only at movies they rated highly
        top_movies = sim_user_ratings[sim_user_ratings > 3.5].index
        for movie in top_movies:
            if movie not in user_seen_movies:
                rec_scores[movie] = rec_scores.get(movie, 0) + sim_user_ratings[movie]

    # Sort by accumulated score
    sorted_recs = sorted(rec_scores.items(), key=lambda x: x[1], reverse=True)[:n_recs]
    return [movie for movie, score in sorted_recs]

# Test Item-Based Recommendations
print("\nItem-Based Recommendations for 'Matrix, The (1999)':")
for i, movie in enumerate(get_item_based_recommendations('Matrix, The (1999)'), 1):
    print(f'{i}. {movie}')

# Test User-Based Recommendations
print("\nUser-Based Recommendations for User ID 1:")
for i, movie in enumerate(get_user_based_recommendations(1), 1):
    print(f'{i}. {movie}')

print()


# Save the optimal model and its dependent data structures
joblib.dump(item_knn, 'item_based_knn_model.pkl')
joblib.dump(movie_user_pivot, 'movie_user_pivot_table.pkl')

print("\n✅ Item-Based Recommender model and Pivot Table successfully exported!\n")


# Tony White ✍️
