# ============================================================
#  FILMIQ — SCRIPT 5: CONTENT-BASED FILTERING
#  File    : python/05_content_based.py
#  Purpose : Recommend movies based on movie features
#            (genres, popularity, quality) rather than
#            user behaviour patterns
#            Solves the cold start problem for new users
#  Run in  : VS Code
# ============================================================

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# STEP 1 — SETUP AND LOAD DATA
# ------------------------------------------------------------
PROJECT_DIR = r'C:\Users\User\Documents\filmiq-movie-recommendation'
os.chdir(PROJECT_DIR)

plt.rcParams.update({
    'figure.facecolor' : 'white',
    'axes.facecolor'   : 'white',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'font.family'      : 'sans-serif',
    'font.size'        : 11
})

BLUE_DARK  = '#0C447C'
BLUE_MID   = '#378ADD'
BLUE_LIGHT = '#B5D4F4'
TEAL       = '#1D9E75'
CORAL      = '#D85A30'

print('Loading data...')
ratings        = pd.read_csv('data/processed/ratings.csv')
movies         = pd.read_csv('data/processed/movies.csv')
train          = pd.read_csv('data/processed/train.csv')
test           = pd.read_csv('data/processed/test.csv')
movie_features = pd.read_csv('data/processed/movie_features.csv',
                              index_col=0)
user_affinity  = pd.read_csv('data/processed/user_affinity.csv',
                              index_col=0)

movie_features.index = movie_features.index.astype(int)
user_affinity.index  = user_affinity.index.astype(int)

print(f'Movie features: {movie_features.shape}')
print(f'User affinity:  {user_affinity.shape}')

# Global mean
global_mean = ratings['rating'].mean()

# ------------------------------------------------------------
# STEP 2 — COMPUTE MOVIE-MOVIE CONTENT SIMILARITY
# Cosine similarity based on genre + popularity + quality
# Two movies are similar if they share genres and have
# comparable quality and popularity scores
# ------------------------------------------------------------
print('\nComputing content similarity matrix...')

content_sim = cosine_similarity(movie_features.values)
content_sim_df = pd.DataFrame(
    content_sim,
    index=movie_features.index,
    columns=movie_features.index
)

print(f'Content similarity matrix: {content_sim_df.shape}')

# Preview — most similar movies to Star Wars
star_wars_id = 50
similar = content_sim_df.loc[star_wars_id].nlargest(6)[1:]
print('\nMovies most similar to Star Wars (content-based):')
for mid, score in similar.items():
    title = movies.loc[movies['movie_id'] == mid, 'title'].values
    title = title[0] if len(title) > 0 else f'Movie {mid}'
    print(f'  {title[:45]:45s} similarity={score:.3f}')

# ------------------------------------------------------------
# STEP 3 — CONTENT-BASED PREDICTION
# Predict how much a user will like a movie based on:
# 1. How similar the movie is to movies they already rated highly
# 2. Their affinity for the genres in that movie
# ------------------------------------------------------------

def predict_rating_content(user_id, movie_id,
                           ratings_df, k=20):
    """
    Predict rating using content-based filtering.

    Steps:
    1. Get movies this user has already rated
    2. Find K movies most similar to movie_id (by content)
    3. Filter to movies the user has actually rated
    4. Weight their ratings by content similarity
    5. Return weighted average
    """
    if movie_id not in content_sim_df.index:
        return global_mean

    # Movies this user has rated
    user_ratings = ratings_df[
        ratings_df['user_id'] == user_id
    ][['movie_id', 'rating']].set_index('movie_id')

    if len(user_ratings) == 0:
        return global_mean

    # Similarities between target movie and all other movies
    similarities = content_sim_df.loc[movie_id].copy()
    similarities[movie_id] = 0  # exclude self

    # Filter to movies user has rated
    rated_movies = user_ratings.index.intersection(
        similarities.index)

    if len(rated_movies) == 0:
        return global_mean

    sim_rated = similarities[rated_movies]
    top_k     = sim_rated.nlargest(k)

    if top_k.sum() == 0:
        return global_mean

    # Weighted average of user's ratings for similar movies
    ratings_k = user_ratings.loc[top_k.index, 'rating']
    predicted  = np.dot(top_k.values,
                        ratings_k.values) / top_k.sum()

    return float(np.clip(predicted, 1, 5))


# Test the function
test_user  = 1
test_movie = 50  # Star Wars
pred = predict_rating_content(test_user, test_movie, train)
actual = ratings[
    (ratings['user_id'] == test_user) &
    (ratings['movie_id'] == test_movie)]['rating'].values
actual = actual[0] if len(actual) > 0 else 'Not rated'
print(f'\nTest — User {test_user}, Star Wars:')
print(f'  Predicted: {pred:.2f} | Actual: {actual}')

# ------------------------------------------------------------
# STEP 4 — EVALUATE CONTENT-BASED MODEL
# ------------------------------------------------------------
print('\nEvaluating content-based filtering...')
print('(sampling 500 test ratings)')

test_sample  = test.sample(500, random_state=42)
preds_cb, actuals_cb = [], []

for _, row in test_sample.iterrows():
    uid  = int(row['user_id'])
    mid  = int(row['movie_id'])
    true = float(row['rating'])
    pred = predict_rating_content(uid, mid, train)
    preds_cb.append(pred)
    actuals_cb.append(true)

preds_cb   = np.array(preds_cb)
actuals_cb = np.array(actuals_cb)

rmse_cb = np.sqrt(mean_squared_error(actuals_cb, preds_cb))
mae_cb  = mean_absolute_error(actuals_cb, preds_cb)

print(f'Content-based results (K=20, n=500):')
print(f'  RMSE: {rmse_cb:.4f}')
print(f'  MAE:  {mae_cb:.4f}')

# ------------------------------------------------------------
# STEP 5 — GENRE-AWARE RECOMMENDATIONS
# Uses user's genre affinity profile to boost/penalise
# movies based on how well their genres match user taste
# ------------------------------------------------------------

def get_content_recommendations(user_id, n=10):
    """
    Generate Top-N content-based recommendations.

    Combines two signals:
    1. Similarity to movies user has rated highly
    2. User's genre affinity profile
    """
    # Movies user has rated
    user_rated = set(ratings[
        ratings['user_id'] == user_id]['movie_id'])

    # Well-rated movies only
    well_rated = set(
        movies[movies['rating_count'] >= 20]['movie_id'])
    candidates = [m for m in movie_features.index
                  if m not in user_rated
                  and m in well_rated]

    # Get user's genre affinity
    if user_id in user_affinity.index:
        affinity = user_affinity.loc[user_id]
        affinity_cols = [c for c in affinity.index
                         if c.endswith('_avg')]
        genre_affinity = affinity[affinity_cols]
    else:
        genre_affinity = None

    # Score each candidate
    scores = []
    for movie_id in candidates:
        # Base score: content similarity prediction
        base_score = predict_rating_content(
            user_id, movie_id, ratings)

        # Genre affinity boost
        if genre_affinity is not None:
            movie_row = movie_features.loc[movie_id]
            genre_cols = ['action', 'adventure', 'animation',
                          'comedy', 'crime', 'drama', 'horror',
                          'romance', 'sci_fi', 'thriller', 'war']
            affinity_boost = 0
            for genre in genre_cols:
                aff_col = f'{genre}_avg'
                if (genre in movie_row.index and
                        aff_col in genre_affinity.index):
                    if movie_row[genre] == 1:
                        affinity_boost += (
                            genre_affinity[aff_col] - global_mean)

            # Small weight on affinity boost
            final_score = base_score + 0.1 * affinity_boost
        else:
            final_score = base_score

        scores.append((movie_id, float(
            np.clip(final_score, 1, 5))))

    scores.sort(key=lambda x: x[1], reverse=True)
    top_n = scores[:n]

    results = []
    for movie_id, score in top_n:
        title = movies.loc[
            movies['movie_id'] == movie_id, 'title'].values
        title = title[0] if len(title) > 0 else f'Movie {movie_id}'
        # Get genres
        row = movie_features.loc[movie_id]
        genre_list = [g.replace('_', '-').title()
                      for g in ['action', 'adventure', 'animation',
                                'comedy', 'crime', 'drama',
                                'film_noir', 'horror', 'romance',
                                'sci_fi', 'thriller', 'war']
                      if g in row.index and row[g] == 1]
        results.append({
            'movie_id'       : movie_id,
            'title'          : title,
            'predicted_score': round(score, 2),
            'genres'         : ', '.join(genre_list[:3])
        })

    return pd.DataFrame(results)


print('\n--- TOP-10 CONTENT-BASED RECOMMENDATIONS FOR USER 1 ---\n')
recs_cb = get_content_recommendations(1, n=10)
print(recs_cb.to_string(index=False))

# ------------------------------------------------------------
# STEP 6 — COLD START DEMONSTRATION
# Content-based filtering works even for new users
# who have only told us their favourite genre
# ------------------------------------------------------------
print('\n--- COLD START DEMO ---')
print('New user says: "I love Sci-Fi films"')
print('Content-based recommendation (no rating history needed):\n')

# Find top Sci-Fi movies by Bayesian average
scifi_movies = movies[
    (movies['sci_fi'] == 1) &
    (movies['rating_count'] >= 20)
].nlargest(10, 'bayesian_avg')[['title', 'bayesian_avg',
                                 'rating_count']]
print(scifi_movies.to_string(index=False))

# ------------------------------------------------------------
# STEP 7 — VISUALISATIONS
# ------------------------------------------------------------

# Chart 1 — content similarity heatmap (sample)
fig, ax = plt.subplots(figsize=(10, 8))
sample_ids = movies.nlargest(20, 'rating_count')['movie_id'].values
sample_sim = content_sim_df.loc[sample_ids, sample_ids]
sample_titles = [
    movies.loc[movies['movie_id'] == mid, 'title'].values[0][:20]
    for mid in sample_ids
]

im = ax.imshow(sample_sim.values, cmap='Blues',
               vmin=0, vmax=1)
ax.set_xticks(range(len(sample_titles)))
ax.set_yticks(range(len(sample_titles)))
ax.set_xticklabels(sample_titles, rotation=45,
                   ha='right', fontsize=7)
ax.set_yticklabels(sample_titles, fontsize=7)
plt.colorbar(im, ax=ax, label='Content similarity')
ax.set_title('Content similarity matrix — top 20 movies\n'
             'Darker = more similar genres and features',
             fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/charts/py12_content_similarity_heatmap.png',
            dpi=150, bbox_inches='tight')
plt.close()
print('\nChart saved: py12_content_similarity_heatmap.png')

# Chart 2 — similar movies to Star Wars (content vs CF)
cf_similar = {
    'Return of the Jedi': 0.681,
    'Raiders of the Lost Ark': 0.612,
    'Empire Strikes Back': 0.598,
    'Indiana Jones': 0.587,
    'Toy Story': 0.543
}

content_similar_ids = content_sim_df.loc[
    star_wars_id].nlargest(6)[1:]
content_similar = {}
for mid, score in content_similar_ids.items():
    title = movies.loc[
        movies['movie_id'] == mid, 'title'].values
    if len(title) > 0:
        content_similar[title[0][:25]] = score

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.barh(list(cf_similar.keys()),
        list(cf_similar.values()),
        color=BLUE_MID, height=0.6)
ax.set_xlabel('Cosine similarity')
ax.set_title('CF similar to Star Wars\n(user behaviour patterns)',
             fontweight='bold')
ax.set_xlim(0, 0.8)

ax2 = axes[1]
ax2.barh(list(content_similar.keys()),
         list(content_similar.values()),
         color=TEAL, height=0.6)
ax2.set_xlabel('Cosine similarity')
ax2.set_title('Content similar to Star Wars\n(genre + features)',
              fontweight='bold')
ax2.set_xlim(0, 1.0)

plt.suptitle('Two ways to find movies similar to Star Wars',
             fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/charts/py13_cf_vs_content_similarity.png',
            dpi=150, bbox_inches='tight')
plt.close()
print('Chart saved: py13_cf_vs_content_similarity.png')

# ------------------------------------------------------------
# STEP 8 — SAVE RESULTS
# ------------------------------------------------------------
cb_results = pd.DataFrame({
    'model'    : ['Content-based filtering'],
    'rmse'     : [rmse_cb],
    'mae'      : [mae_cb],
    'test_size': [len(preds_cb)]
})

cb_results.to_csv('data/processed/cb_results.csv', index=False)
recs_cb.to_csv('data/processed/user1_recs_cb.csv', index=False)

print('\n--- CONTENT-BASED MODEL SUMMARY ---')
print(cb_results.to_string(index=False))
print('\nScript 05 complete.')
