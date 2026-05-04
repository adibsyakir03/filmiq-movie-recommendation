# ============================================================
#  FILMIQ — SCRIPT 3: COLLABORATIVE FILTERING
#  File    : python/03_collaborative_filtering.py
#  Purpose : Build user-based and item-based collaborative
#            filtering models using cosine similarity
#            Predict ratings and generate Top-10 recommendations
#  Run in  : VS Code
# ============================================================

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
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
TEAL       = '#1D9E75'
CORAL      = '#D85A30'

print('Loading feature matrices...')
ratings     = pd.read_csv('data/processed/ratings.csv')
movies      = pd.read_csv('data/processed/movies.csv')
train       = pd.read_csv('data/processed/train.csv')
test        = pd.read_csv('data/processed/test.csv')
user_item   = pd.read_csv('data/processed/user_item_matrix.csv',
                           index_col=0)
user_item.index   = user_item.index.astype(int)
user_item.columns = user_item.columns.astype(int)

print(f'User-item matrix: {user_item.shape}')
print(f'Train: {len(train):,} | Test: {len(test):,}')

# ------------------------------------------------------------
# STEP 2 — COMPUTE USER SIMILARITY MATRIX
# Cosine similarity between every pair of users
# Based on their rating vectors in the user-item matrix
# ------------------------------------------------------------
print('\nComputing user similarity matrix...')

user_matrix = user_item.values.astype(float)
user_sim    = cosine_similarity(user_matrix)
user_sim_df = pd.DataFrame(
    user_sim,
    index=user_item.index,
    columns=user_item.index
)

print(f'User similarity matrix: {user_sim_df.shape}')
print(f'Average similarity between users: '
      f'{user_sim[np.triu_indices_from(user_sim, k=1)].mean():.4f}')

# ------------------------------------------------------------
# STEP 3 — USER-BASED CF: PREDICT A RATING
# For a given user and movie, find the K most similar users
# who have rated that movie, then take their weighted average
# ------------------------------------------------------------

def predict_rating_user_based(user_id, movie_id, k=20):
    """
    Predict rating for user_id on movie_id using K nearest neighbours.

    Steps:
    1. Find K users most similar to user_id who rated movie_id
    2. Weight their ratings by similarity score
    3. Return weighted average as predicted rating
    """
    if movie_id not in user_item.columns:
        return user_item.loc[user_id].replace(0, np.nan).mean()

    # Get similarities for this user
    similarities = user_sim_df.loc[user_id].copy()
    similarities[user_id] = 0  # exclude self

    # Get users who rated this movie
    movie_ratings = user_item[movie_id]
    rated_users   = movie_ratings[movie_ratings > 0].index

    if len(rated_users) == 0:
        return 3.53  # global mean fallback

    # Filter to users who rated this movie
    sim_rated = similarities[rated_users]

    # Take top K most similar
    top_k = sim_rated.nlargest(k)

    if top_k.sum() == 0:
        return 3.53  # fallback if no similar users

    # Weighted average
    ratings_k = movie_ratings[top_k.index]
    predicted  = np.dot(top_k.values, ratings_k.values) / top_k.sum()

    # Clip to valid rating range
    return float(np.clip(predicted, 1, 5))


# Test the prediction function
test_user  = 1
test_movie = 50  # Star Wars
pred = predict_rating_user_based(test_user, test_movie)
actual = user_item.loc[test_user, test_movie]
print(f'\nTest prediction — User {test_user}, '
      f'Star Wars (movie 50):')
print(f'  Predicted: {pred:.2f} | Actual: {actual:.1f}')

# ------------------------------------------------------------
# STEP 4 — EVALUATE USER-BASED CF ON TEST SET
# Run predictions on a sample of test ratings
# (full test set would take too long for K-NN)
# ------------------------------------------------------------
print('\nEvaluating user-based CF on test set...')
print('(sampling 500 test ratings for speed)')

test_sample = test.sample(500, random_state=42)
predictions = []
actuals     = []

for _, row in test_sample.iterrows():
    uid  = int(row['user_id'])
    mid  = int(row['movie_id'])
    true = float(row['rating'])

    if uid in user_item.index and mid in user_item.columns:
        pred = predict_rating_user_based(uid, mid, k=20)
        predictions.append(pred)
        actuals.append(true)

predictions = np.array(predictions)
actuals     = np.array(actuals)

rmse_user = np.sqrt(mean_squared_error(actuals, predictions))
mae_user  = mean_absolute_error(actuals, predictions)

print(f'User-based CF results (K=20, n=500):')
print(f'  RMSE: {rmse_user:.4f}')
print(f'  MAE:  {mae_user:.4f}')

# ------------------------------------------------------------
# STEP 5 — COMPUTE ITEM SIMILARITY MATRIX
# Cosine similarity between every pair of movies
# Based on their rating vectors (who rated them and how)
# ------------------------------------------------------------
print('\nComputing item similarity matrix...')

item_matrix = user_item.values.T.astype(float)
item_sim    = cosine_similarity(item_matrix)
item_sim_df = pd.DataFrame(
    item_sim,
    index=user_item.columns,
    columns=user_item.columns
)

print(f'Item similarity matrix: {item_sim_df.shape}')

# ------------------------------------------------------------
# STEP 6 — ITEM-BASED CF: PREDICT A RATING
# ------------------------------------------------------------

def predict_rating_item_based(user_id, movie_id, k=20):
    """
    Predict rating using item-based collaborative filtering.

    Steps:
    1. Find K movies most similar to movie_id
    2. Filter to movies the user has actually rated
    3. Weight those ratings by item similarity
    4. Return weighted average
    """
    if movie_id not in item_sim_df.index:
        return 3.53

    # Get similarities for this movie
    similarities = item_sim_df.loc[movie_id].copy()
    similarities[movie_id] = 0  # exclude self

    # Get movies this user has rated
    user_ratings  = user_item.loc[user_id]
    rated_movies  = user_ratings[user_ratings > 0].index

    if len(rated_movies) == 0:
        return 3.53

    # Filter to rated movies
    sim_rated = similarities[rated_movies]
    top_k     = sim_rated.nlargest(k)

    if top_k.sum() == 0:
        return 3.53

    ratings_k = user_ratings[top_k.index]
    predicted  = np.dot(top_k.values, ratings_k.values) / top_k.sum()

    return float(np.clip(predicted, 1, 5))


# Evaluate item-based CF
print('\nEvaluating item-based CF on test set...')
predictions_item = []
actuals_item     = []

for _, row in test_sample.iterrows():
    uid  = int(row['user_id'])
    mid  = int(row['movie_id'])
    true = float(row['rating'])

    if uid in user_item.index and mid in user_item.columns:
        pred = predict_rating_item_based(uid, mid, k=20)
        predictions_item.append(pred)
        actuals_item.append(true)

predictions_item = np.array(predictions_item)
actuals_item     = np.array(actuals_item)

rmse_item = np.sqrt(mean_squared_error(
    actuals_item, predictions_item))
mae_item  = mean_absolute_error(actuals_item, predictions_item)

print(f'Item-based CF results (K=20, n=500):')
print(f'  RMSE: {rmse_item:.4f}')
print(f'  MAE:  {mae_item:.4f}')

# ------------------------------------------------------------
# STEP 7 — GENERATE TOP-10 RECOMMENDATIONS
# For a given user, find movies they haven't seen
# that are predicted to score highest
# ------------------------------------------------------------

def get_top_n_recommendations(user_id, n=10, method='user'):
    """
    Generate Top-N movie recommendations for a user.
    method: 'user' for user-based CF, 'item' for item-based CF
    """
    # Movies the user has already rated
    rated = set(user_item.loc[user_id][
        user_item.loc[user_id] > 0].index)

    # All movies not yet rated
    well_rated = set(movies[movies['rating_count'] >= 20]['movie_id'])
    unrated = [m for m in user_item.columns 
           if m not in rated and m in well_rated]

    # Predict ratings for all unrated movies
    preds = []
    for movie_id in unrated:
        if method == 'user':
            pred = predict_rating_user_based(user_id, movie_id)
        else:
            pred = predict_rating_item_based(user_id, movie_id)
        preds.append((movie_id, pred))

    # Sort by predicted rating
    preds.sort(key=lambda x: x[1], reverse=True)
    top_n = preds[:n]

    # Add movie titles
    results = []
    for movie_id, pred_rating in top_n:
        title = movies.loc[
            movies['movie_id'] == movie_id, 'title'].values
        title = title[0] if len(title) > 0 else f'Movie {movie_id}'
        results.append({
            'movie_id'        : movie_id,
            'title'           : title,
            'predicted_rating': round(pred_rating, 2)
        })

    return pd.DataFrame(results)


# Demo — recommendations for User 1
print('\n--- TOP-10 RECOMMENDATIONS FOR USER 1 ---')
print('(User-based collaborative filtering)\n')
recs_user = get_top_n_recommendations(1, n=10, method='user')
print(recs_user.to_string(index=False))

print('\n(Item-based collaborative filtering)\n')
recs_item = get_top_n_recommendations(1, n=10, method='item')
print(recs_item.to_string(index=False))

# What has User 1 already rated highly?
print('\n--- USER 1 TOP-RATED MOVIES (actual) ---')
user1_ratings = ratings[ratings['user_id'] == 1].merge(
    movies[['movie_id', 'title']], on='movie_id')
print(user1_ratings.nlargest(10, 'rating')[
    ['title', 'rating']].to_string(index=False))

# ------------------------------------------------------------
# STEP 8 — VISUALISATION
# ------------------------------------------------------------

# Chart 1 — prediction error distribution
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
errors_user = predictions - actuals
ax.hist(errors_user, bins=20, color=BLUE_MID,
        edgecolor='white', linewidth=0.8, zorder=3)
ax.axvline(x=0, color=CORAL, linewidth=2, linestyle='--')
ax.set_xlabel('Prediction error (predicted - actual)')
ax.set_ylabel('Count')
ax.set_title(f'User-based CF prediction errors\n'
             f'RMSE={rmse_user:.3f}, MAE={mae_user:.3f}',
             fontweight='bold')

ax2 = axes[1]
errors_item = predictions_item - actuals_item
ax2.hist(errors_item, bins=20, color=TEAL,
         edgecolor='white', linewidth=0.8, zorder=3)
ax2.axvline(x=0, color=CORAL, linewidth=2, linestyle='--')
ax2.set_xlabel('Prediction error (predicted - actual)')
ax2.set_ylabel('Count')
ax2.set_title(f'Item-based CF prediction errors\n'
              f'RMSE={rmse_item:.3f}, MAE={mae_item:.3f}',
              fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/charts/py07_cf_prediction_errors.png',
            dpi=150, bbox_inches='tight')
plt.close()
print('\nChart saved: py07_cf_prediction_errors.png')

# Chart 2 — similar movies to Star Wars
print('\nFinding movies similar to Star Wars...')
star_wars_id  = 50
similar_movies = item_sim_df.loc[star_wars_id].nlargest(11)[1:]
similar_titles = []
for mid in similar_movies.index:
    title = movies.loc[
        movies['movie_id'] == mid, 'title'].values
    title = title[0] if len(title) > 0 else f'Movie {mid}'
    similar_titles.append(title[:30])

fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.barh(similar_titles[::-1],
               similar_movies.values[::-1],
               color=BLUE_MID, height=0.7, zorder=3)
for bar, val in zip(bars, similar_movies.values[::-1]):
    ax.text(val + 0.002,
            bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9,
            color=BLUE_DARK)
ax.set_xlabel('Cosine similarity score')
ax.set_title('Movies most similar to Star Wars (1977)\n'
             'Based on shared user rating patterns',
             fontweight='bold')
ax.set_xlim(0, 0.75)
plt.tight_layout()
plt.savefig('outputs/charts/py08_similar_to_starwars.png',
            dpi=150, bbox_inches='tight')
plt.close()
print('Chart saved: py08_similar_to_starwars.png')

# ------------------------------------------------------------
# STEP 9 — SAVE RESULTS
# ------------------------------------------------------------
cf_results = pd.DataFrame({
    'model'     : ['User-based CF', 'Item-based CF'],
    'rmse'      : [rmse_user, rmse_item],
    'mae'       : [mae_user, mae_item],
    'k_neighbours': [20, 20],
    'test_size' : [len(predictions), len(predictions_item)]
})

cf_results.to_csv('data/processed/cf_results.csv', index=False)
recs_user.to_csv('data/processed/user1_recs_usercf.csv',
                 index=False)
recs_item.to_csv('data/processed/user1_recs_itemcf.csv',
                 index=False)

print('\n--- CF MODEL SUMMARY ---')
print(cf_results.to_string(index=False))
print('\nScript 03 complete.')
