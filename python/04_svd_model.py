# ============================================================
#  FILMIQ — SCRIPT 4: SVD MATRIX FACTORISATION
#  File    : python/04_svd_model.py
#  Purpose : Implement SVD (Singular Value Decomposition)
#            the method that won the Netflix Prize in 2009
#            Decomposes user-item matrix into latent factors
#            representing hidden user preferences and movie traits
#  Run in  : VS Code
# ============================================================

import pandas as pd
import numpy as np
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
BLUE_LIGHT = '#B5D4F4'
TEAL       = '#1D9E75'
CORAL      = '#D85A30'
GRAY       = '#888780'

print('Loading data...')
ratings  = pd.read_csv('data/processed/ratings.csv')
movies   = pd.read_csv('data/processed/movies.csv')
train    = pd.read_csv('data/processed/train.csv')
test     = pd.read_csv('data/processed/test.csv')
user_item = pd.read_csv('data/processed/user_item_matrix.csv',
                         index_col=0)
user_item.index   = user_item.index.astype(int)
user_item.columns = user_item.columns.astype(int)

print(f'User-item matrix: {user_item.shape}')

# Global mean — used as fallback and for bias correction
global_mean = ratings['rating'].mean()
print(f'Global mean rating: {global_mean:.4f}')

# ------------------------------------------------------------
# STEP 2 — MEAN-CENTER THE MATRIX
# Subtract global mean + user bias + item bias
# This is the key improvement over basic SVD
# (called SVD++ or biased SVD in the literature)
# ------------------------------------------------------------
print('\nMean-centering the matrix...')

# User bias — how much each user rates above/below global mean
user_bias = ratings.groupby('user_id')['rating'].mean() - global_mean

# Item bias — how much each movie is rated above/below global mean
item_bias = ratings.groupby('movie_id')['rating'].mean() - global_mean

# Build centered matrix
R = user_item.values.astype(float)

# For rated entries only, subtract global mean + biases
for i, user_id in enumerate(user_item.index):
    for j, movie_id in enumerate(user_item.columns):
        if R[i, j] > 0:
            u_bias = user_bias.get(user_id, 0)
            m_bias = item_bias.get(movie_id, 0)
            R[i, j] -= (global_mean + u_bias + m_bias)

print('Matrix centered on global mean + user/item biases')

# ------------------------------------------------------------
# STEP 3 — APPLY SVD
# Decompose R into U × Σ × V^T
# U = user latent factors (how much each user "is" each factor)
# Σ = singular values (importance of each factor)
# V = item latent factors (how much each movie "has" each factor)
#
# Think of factors as hidden concepts like:
# factor 1 = "serious drama", factor 2 = "action/adventure"
# factor 3 = "comedy" etc. — the model learns these automatically
# ------------------------------------------------------------
print('\nApplying SVD decomposition...')

U, sigma, Vt = np.linalg.svd(R, full_matrices=False)

print(f'U shape (users × factors):   {U.shape}')
print(f'Sigma shape (factors):        {sigma.shape}')
print(f'Vt shape (factors × movies):  {Vt.shape}')
print(f'Total factors available:      {len(sigma)}')

# Explained variance by number of factors
total_variance = np.sum(sigma ** 2)
cumulative_var = np.cumsum(sigma ** 2) / total_variance

# Find how many factors explain 90% of variance
n_factors_90 = np.argmax(cumulative_var >= 0.90) + 1
n_factors_95 = np.argmax(cumulative_var >= 0.95) + 1
print(f'\nFactors needed to explain 90% variance: {n_factors_90}')
print(f'Factors needed to explain 95% variance: {n_factors_95}')

# ------------------------------------------------------------
# STEP 4 — RECONSTRUCT MATRIX WITH K FACTORS
# Use only the top K factors to reconstruct predictions
# More factors = more accurate but more overfitting
# ------------------------------------------------------------

def reconstruct_matrix(U, sigma, Vt, k):
    """Reconstruct rating matrix using top k latent factors."""
    U_k     = U[:, :k]
    sigma_k = np.diag(sigma[:k])
    Vt_k    = Vt[:k, :]
    return U_k @ sigma_k @ Vt_k


def predict_svd(user_id, movie_id, R_pred,
                user_index, movie_index):
    """
    Predict rating for a user-movie pair from reconstructed matrix.
    Add back the global mean and biases removed during centering.
    """
    if user_id not in user_index or movie_id not in movie_index:
        return global_mean

    i = user_index[user_id]
    j = movie_index[movie_id]

    # Base prediction from SVD
    pred = R_pred[i, j]

    # Add back biases
    pred += global_mean
    pred += user_bias.get(user_id, 0)
    pred += item_bias.get(movie_id, 0)

    return float(np.clip(pred, 1, 5))


# Index mappings
user_index  = {uid: i for i, uid in enumerate(user_item.index)}
movie_index = {mid: j for j, mid in enumerate(user_item.columns)}

# ------------------------------------------------------------
# STEP 5 — FIND OPTIMAL K (NUMBER OF FACTORS)
# Test multiple values of K and find the one with lowest RMSE
# ------------------------------------------------------------
print('\nFinding optimal number of latent factors...')

test_sample = test.sample(min(1000, len(test)), random_state=42)
k_values = [5, 10, 20, 30, 50, 75, 100]
rmse_by_k   = []

for k in k_values:
    R_pred = reconstruct_matrix(U, sigma, Vt, k)
    preds, actuals = [], []

    for _, row in test_sample.iterrows():
        uid  = int(row['user_id'])
        mid  = int(row['movie_id'])
        true = float(row['rating'])
        pred = predict_svd(uid, mid, R_pred,
                           user_index, movie_index)
        preds.append(pred)
        actuals.append(true)

    rmse = np.sqrt(mean_squared_error(actuals, preds))
    rmse_by_k.append(rmse)
    print(f'  K={k:4d}: RMSE={rmse:.4f}')

best_k    = k_values[np.argmin(rmse_by_k)]
best_rmse = min(rmse_by_k)
print(f'\nBest K: {best_k} (RMSE={best_rmse:.4f})')

# ------------------------------------------------------------
# STEP 6 — EVALUATE BEST MODEL
# ------------------------------------------------------------
print(f'\nEvaluating SVD with K={best_k}...')

R_pred_best = reconstruct_matrix(U, sigma, Vt, best_k)

test_sample_large = test.sample(500, random_state=42)
preds_svd, actuals_svd = [], []

for _, row in test_sample_large.iterrows():
    uid  = int(row['user_id'])
    mid  = int(row['movie_id'])
    true = float(row['rating'])
    pred = predict_svd(uid, mid, R_pred_best,
                       user_index, movie_index)
    preds_svd.append(pred)
    actuals_svd.append(true)

preds_svd   = np.array(preds_svd)
actuals_svd = np.array(actuals_svd)

rmse_svd = np.sqrt(mean_squared_error(actuals_svd, preds_svd))
mae_svd  = mean_absolute_error(actuals_svd, preds_svd)

print(f'SVD results (K={best_k}, n=500):')
print(f'  RMSE: {rmse_svd:.4f}')
print(f'  MAE:  {mae_svd:.4f}')

# ------------------------------------------------------------
# STEP 7 — SVD RECOMMENDATIONS FOR USER 1
# ------------------------------------------------------------
print('\n--- TOP-10 SVD RECOMMENDATIONS FOR USER 1 ---\n')

user_id = 1
rated   = set(user_item.loc[user_id][
    user_item.loc[user_id] > 0].index)
well_rated_movies = set(
    movies[movies['rating_count'] >= 20]['movie_id'])
unrated = [m for m in user_item.columns
           if m not in rated and m in well_rated_movies]

svd_preds = []
for movie_id in unrated:
    pred = predict_svd(user_id, movie_id, R_pred_best,
                       user_index, movie_index)
    svd_preds.append((movie_id, pred))

svd_preds.sort(key=lambda x: x[1], reverse=True)
top10_svd = svd_preds[:10]

recs_svd = []
for movie_id, pred in top10_svd:
    title = movies.loc[
        movies['movie_id'] == movie_id, 'title'].values
    title = title[0] if len(title) > 0 else f'Movie {movie_id}'
    recs_svd.append({
        'movie_id'        : movie_id,
        'title'           : title,
        'predicted_rating': round(pred, 2)
    })

recs_svd_df = pd.DataFrame(recs_svd)
print(recs_svd_df.to_string(index=False))

# ------------------------------------------------------------
# STEP 8 — EXPLORE LATENT FACTORS
# What do the top factors represent?
# ------------------------------------------------------------
print('\n--- LATENT FACTOR ANALYSIS ---')
print(f'Using K={best_k} factors\n')

# For each of the first 5 factors, find movies that score highest
Vt_best = Vt[:best_k, :]

for factor_idx in range(min(3, best_k)):
    factor_scores = Vt_best[factor_idx, :]
    top_movie_indices = np.argsort(factor_scores)[::-1][:5]
    top_movie_ids = [user_item.columns[i]
                     for i in top_movie_indices]

    print(f'Factor {factor_idx + 1} — top movies:')
    for mid in top_movie_ids:
        title = movies.loc[
            movies['movie_id'] == mid, 'title'].values
        title = title[0] if len(title) > 0 else f'Movie {mid}'
        score = factor_scores[list(
            user_item.columns).index(mid)]
        print(f'  {title[:40]:40s} score={score:.3f}')
    print()

# ------------------------------------------------------------
# STEP 9 — VISUALISATIONS
# ------------------------------------------------------------

# Chart 1 — RMSE by number of factors
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(k_values, rmse_by_k, color=BLUE_DARK,
        linewidth=2.5, marker='o', markersize=8, zorder=3)
ax.axvline(x=best_k, color=CORAL, linewidth=1.5,
           linestyle='--',
           label=f'Best K={best_k} (RMSE={best_rmse:.3f})')
for k, rmse in zip(k_values, rmse_by_k):
    ax.annotate(f'{rmse:.3f}',
                xy=(k, rmse),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center', fontsize=9, color=BLUE_DARK)
ax.set_xlabel('Number of latent factors (K)')
ax.set_ylabel('RMSE')
ax.set_title('SVD model accuracy by number of latent factors\n'
             'Finding the optimal K',
             fontweight='bold')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('outputs/charts/py09_svd_factors_rmse.png',
            dpi=150, bbox_inches='tight')
plt.close()
print('Chart saved: py09_svd_factors_rmse.png')

# Chart 2 — explained variance
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, min(101, len(cumulative_var) + 1)),
        cumulative_var[:100] * 100,
        color=BLUE_DARK, linewidth=2.5, zorder=3)
ax.axhline(y=90, color=CORAL, linewidth=1.5,
           linestyle='--', label='90% variance')
ax.axhline(y=95, color=TEAL, linewidth=1.5,
           linestyle='--', label='95% variance')
ax.fill_between(range(1, min(101, len(cumulative_var) + 1)),
                cumulative_var[:100] * 100,
                alpha=0.15, color=BLUE_MID)
ax.set_xlabel('Number of latent factors')
ax.set_ylabel('Cumulative explained variance (%)')
ax.set_title('SVD explained variance by number of factors\n'
             'How many factors capture the rating patterns',
             fontweight='bold')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('outputs/charts/py10_svd_explained_variance.png',
            dpi=150, bbox_inches='tight')
plt.close()
print('Chart saved: py10_svd_explained_variance.png')

# Chart 3 — SVD prediction errors
fig, ax = plt.subplots(figsize=(10, 5))
errors_svd = preds_svd - actuals_svd
ax.hist(errors_svd, bins=25, color=TEAL,
        edgecolor='white', linewidth=0.8, zorder=3)
ax.axvline(x=0, color=CORAL, linewidth=2, linestyle='--')
ax.set_xlabel('Prediction error (predicted - actual)')
ax.set_ylabel('Count')
ax.set_title(f'SVD prediction errors (K={best_k})\n'
             f'RMSE={rmse_svd:.3f}, MAE={mae_svd:.3f}',
             fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/charts/py11_svd_prediction_errors.png',
            dpi=150, bbox_inches='tight')
plt.close()
print('Chart saved: py11_svd_prediction_errors.png')

# ------------------------------------------------------------
# STEP 10 — SAVE RESULTS
# ------------------------------------------------------------
svd_results = pd.DataFrame({
    'model'    : [f'SVD (K={best_k})'],
    'rmse'     : [rmse_svd],
    'mae'      : [mae_svd],
    'k_factors': [best_k],
    'test_size': [len(preds_svd)]
})

svd_results.to_csv('data/processed/svd_results.csv', index=False)
recs_svd_df.to_csv('data/processed/user1_recs_svd.csv',
                   index=False)

print('\n--- SVD MODEL SUMMARY ---')
print(svd_results.to_string(index=False))
print('\nScript 04 complete.')
