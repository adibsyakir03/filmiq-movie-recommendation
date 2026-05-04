# ============================================================
#  FILMIQ — SCRIPT 2: FEATURE ENGINEERING
#  File    : python/02_features.py
#  Purpose : Build feature matrices for all three models:
#            - User-item matrix for collaborative filtering
#            - Genre vectors for content-based filtering
#            - User genre affinity profiles
#  Run in  : VS Code
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine
import os
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# STEP 1 — SETUP
# ------------------------------------------------------------
PROJECT_DIR = r'C:\Users\User\Documents\filmiq-movie-recommendation'
os.chdir(PROJECT_DIR)

# Update password if needed
engine = create_engine('mysql+pymysql://root:Actuary123@localhost/filmiq')

print('Loading data...')
ratings  = pd.read_csv('data/processed/ratings.csv')
movies   = pd.read_csv('data/processed/movies.csv')
users    = pd.read_csv('data/processed/users.csv')
print(f'Loaded: {len(ratings):,} ratings, '
      f'{len(movies):,} movies, {len(users):,} users')

# ------------------------------------------------------------
# STEP 2 — USER-ITEM MATRIX
# Rows = users, columns = movies, values = ratings
# This is the core data structure for collaborative filtering
# ------------------------------------------------------------
print('\nBuilding user-item matrix...')

user_item = ratings.pivot_table(
    index='user_id',
    columns='movie_id',
    values='rating',
    fill_value=0
)

print(f'User-item matrix shape: {user_item.shape}')
print(f'Matrix density: '
      f'{(user_item > 0).sum().sum() / user_item.size * 100:.2f}%')

# Mean-centered user-item matrix
# Subtract each user's average rating to remove rating bias
user_means = ratings.groupby('user_id')['rating'].mean()
user_item_centered = user_item.copy().astype(float)

for user_id in user_item_centered.index:
    mask = user_item_centered.loc[user_id] > 0
    user_item_centered.loc[user_id, mask] -= user_means[user_id]

print('Mean-centered matrix built')

# Save
user_item.to_csv('data/processed/user_item_matrix.csv')
user_item_centered.to_csv(
    'data/processed/user_item_centered.csv')
print('User-item matrices saved')

# ------------------------------------------------------------
# STEP 3 — GENRE FEATURE VECTORS FOR MOVIES
# Each movie represented as a binary vector of 19 genres
# This is the input to content-based filtering
# ------------------------------------------------------------
print('\nBuilding genre feature vectors...')

genre_cols = [
    'action', 'adventure', 'animation', 'children',
    'comedy', 'crime', 'documentary', 'drama',
    'fantasy', 'film_noir', 'horror', 'musical',
    'mystery', 'romance', 'sci_fi', 'thriller',
    'war', 'western'
]

# Extract genre matrix
genre_matrix = movies[['movie_id', 'title'] + genre_cols].copy()
genre_matrix = genre_matrix.set_index('movie_id')

# Fill any NaN genre flags with 0
genre_matrix[genre_cols] = genre_matrix[genre_cols].fillna(0)

print(f'Genre matrix shape: {genre_matrix[genre_cols].shape}')
print(f'Movies with no genre: '
      f'{(genre_matrix[genre_cols].sum(axis=1) == 0).sum()}')

# Combined feature matrix — genre + popularity + quality
# Normalise continuous features to 0-1 scale
scaler = MinMaxScaler()
movie_features = genre_matrix[genre_cols].copy().astype(float)

# Add normalised popularity and quality scores
movie_features['popularity_norm'] = scaler.fit_transform(
    movies[['popularity_score']].fillna(0)
)
movie_features['quality_norm'] = scaler.fit_transform(
    movies[['bayesian_avg']].fillna(movies['bayesian_avg'].mean())
)

print(f'Movie feature matrix shape: {movie_features.shape}')
print(f'Features per movie: {movie_features.shape[1]}')
print('  - 18 genre binary flags')
print('  - 1 normalised popularity score')
print('  - 1 normalised quality score (Bayesian avg)')

movie_features.to_csv('data/processed/movie_features.csv')
genre_matrix.to_csv('data/processed/genre_matrix.csv')
print('Movie feature matrices saved')

# ------------------------------------------------------------
# STEP 4 — USER GENRE AFFINITY PROFILES
# For each user, calculate their average rating per genre
# This is their "taste fingerprint" for content-based filtering
# ------------------------------------------------------------
print('\nBuilding user genre affinity profiles...')

# Merge ratings with genre flags
merged = ratings.merge(
    movies[['movie_id'] + genre_cols],
    on='movie_id'
)

# Calculate mean rating per genre per user
affinity_data = []
for genre in genre_cols:
    genre_ratings = merged[merged[genre] == 1].groupby(
        'user_id')['rating'].agg(['mean', 'count']).reset_index()
    genre_ratings.columns = ['user_id', f'{genre}_avg',
                              f'{genre}_count']
    affinity_data.append(genre_ratings.set_index('user_id'))

# Combine all genre affinities
user_affinity = pd.concat(affinity_data, axis=1)

# Fill missing genre affinities with global mean (3.53)
avg_cols = [c for c in user_affinity.columns
            if c.endswith('_avg')]
user_affinity[avg_cols] = user_affinity[avg_cols].fillna(3.53)

cnt_cols = [c for c in user_affinity.columns
            if c.endswith('_count')]
user_affinity[cnt_cols] = user_affinity[cnt_cols].fillna(0)

print(f'User affinity profile shape: {user_affinity.shape}')
print(f'Genre affinity features: {len(avg_cols)}')

user_affinity.to_csv('data/processed/user_affinity.csv')
print('User affinity profiles saved')

# ------------------------------------------------------------
# STEP 5 — TRAIN/TEST SPLIT
# Use random split instead of timestamp-based
# MovieLens timestamps are too clustered for chronological split
print('\nCreating train/test split...')

from sklearn.model_selection import train_test_split

train, test = train_test_split(
    ratings,
    test_size=0.2,
    random_state=42,
    stratify=ratings['user_id']  # ensure each user appears in both
)

print(f'Train set: {len(train):,} ratings')
print(f'Test set:  {len(test):,} ratings')

# Filter cold start from test
train_users  = set(train['user_id'])
train_movies = set(train['movie_id'])
test_filtered = test[
    test['user_id'].isin(train_users) &
    test['movie_id'].isin(train_movies)
]

print(f'Test set after filtering: {len(test_filtered):,} ratings')

train.to_csv('data/processed/train.csv', index=False)
test_filtered.to_csv('data/processed/test.csv', index=False)
print('Train/test split saved')

# ------------------------------------------------------------
# STEP 6 — FEATURE SUMMARY
# ------------------------------------------------------------
print('\n--- FEATURE ENGINEERING SUMMARY ---')
print(f'User-item matrix:     {user_item.shape[0]} users × '
      f'{user_item.shape[1]} movies')
print(f'Movie features:       {movie_features.shape[0]} movies × '
      f'{movie_features.shape[1]} features')
print(f'User affinity:        {user_affinity.shape[0]} users × '
      f'{len(avg_cols)} genre scores')
print(f'Training ratings:     {len(train):,}')
print(f'Test ratings:         {len(test_filtered):,}')
print(f'\nAll features saved to data/processed/')
print('Script 02 complete.')
