# ============================================================
#  FILMIQ — SCRIPT 6: HYBRID RECOMMENDATION MODEL
#  File    : python/06_hybrid_model.py
#  Purpose : Combine SVD + Item-based CF + Content-based
#            filtering into a single hybrid recommender
#            Adaptively weights models based on user activity
#            Produces final Top-10 recommendations per user
#  Run in  : VS Code
# ============================================================

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# STEP 1 — SETUP AND LOAD ALL MODEL OUTPUTS
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
AMBER      = '#BA7517'
GRAY       = '#888780'

print('Loading all model components...')

ratings        = pd.read_csv('data/processed/ratings.csv')
movies         = pd.read_csv('data/processed/movies.csv')
train          = pd.read_csv('data/processed/train.csv')
test           = pd.read_csv('data/processed/test.csv')
user_item      = pd.read_csv('data/processed/user_item_matrix.csv',
                              index_col=0)
movie_features = pd.read_csv('data/processed/movie_features.csv',
                              index_col=0)
user_affinity  = pd.read_csv('data/processed/user_affinity.csv',
                              index_col=0)

user_item.index      = user_item.index.astype(int)
user_item.columns    = user_item.columns.astype(int)
movie_features.index = movie_features.index.astype(int)
user_affinity.index  = user_affinity.index.astype(int)

global_mean = ratings['rating'].mean()
user_bias   = ratings.groupby('user_id')['rating'].mean() - global_mean
item_bias   = ratings.groupby('movie_id')['rating'].mean() - global_mean

print('Computing similarity matrices...')

# User similarity (for CF)
user_sim = cosine_similarity(user_item.values.astype(float))
user_sim_df = pd.DataFrame(user_sim,
                            index=user_item.index,
                            columns=user_item.index)

# Item similarity (for CF)
item_sim = cosine_similarity(user_item.values.T.astype(float))
item_sim_df = pd.DataFrame(item_sim,
                            index=user_item.columns,
                            columns=user_item.columns)

# Content similarity
content_sim = cosine_similarity(movie_features.values)
content_sim_df = pd.DataFrame(content_sim,
                               index=movie_features.index,
                               columns=movie_features.index)

# SVD decomposition
print('Running SVD...')
R = user_item.values.astype(float)
for i, uid in enumerate(user_item.index):
    for j, mid in enumerate(user_item.columns):
        if R[i, j] > 0:
            R[i, j] -= (global_mean +
                        user_bias.get(uid, 0) +
                        item_bias.get(mid, 0))

U, sigma, Vt = np.linalg.svd(R, full_matrices=False)
K = 100
U_k  = U[:, :K]
S_k  = np.diag(sigma[:K])
Vt_k = Vt[:K, :]
R_pred = U_k @ S_k @ Vt_k

user_index  = {uid: i for i, uid in enumerate(user_item.index)}
movie_index = {mid: j for j, mid in enumerate(user_item.columns)}

print('All components ready.')

# ------------------------------------------------------------
# STEP 2 — INDIVIDUAL PREDICTION FUNCTIONS
# Compact versions of each model's prediction
# ------------------------------------------------------------

def pred_svd(user_id, movie_id):
    if user_id not in user_index or movie_id not in movie_index:
        return global_mean
    i, j = user_index[user_id], movie_index[movie_id]
    pred  = R_pred[i, j]
    pred += global_mean
    pred += user_bias.get(user_id, 0)
    pred += item_bias.get(movie_id, 0)
    return float(np.clip(pred, 1, 5))


def pred_item_cf(user_id, movie_id, k=20):
    if movie_id not in item_sim_df.index:
        return global_mean
    similarities = item_sim_df.loc[movie_id].copy()
    similarities[movie_id] = 0
    user_ratings = user_item.loc[user_id]
    rated        = user_ratings[user_ratings > 0].index
    if len(rated) == 0:
        return global_mean
    sim_rated = similarities[rated].nlargest(k)
    if sim_rated.sum() == 0:
        return global_mean
    return float(np.clip(
        np.dot(sim_rated.values,
               user_ratings[sim_rated.index].values)
        / sim_rated.sum(), 1, 5))


def pred_content(user_id, movie_id, k=20):
    if movie_id not in content_sim_df.index:
        return global_mean
    user_ratings = ratings[ratings['user_id'] == user_id][
        ['movie_id', 'rating']].set_index('movie_id')
    if len(user_ratings) == 0:
        return global_mean
    similarities = content_sim_df.loc[movie_id].copy()
    similarities[movie_id] = 0
    rated     = user_ratings.index.intersection(similarities.index)
    if len(rated) == 0:
        return global_mean
    sim_rated = similarities[rated].nlargest(k)
    if sim_rated.sum() == 0:
        return global_mean
    return float(np.clip(
        np.dot(sim_rated.values,
               user_ratings.loc[sim_rated.index,
                                'rating'].values)
        / sim_rated.sum(), 1, 5))


# ------------------------------------------------------------
# STEP 3 — ADAPTIVE HYBRID PREDICTION
# Weights adjust based on user activity level:
#
# Active users (100+ ratings):   SVD=0.6, CF=0.3, CB=0.1
# Moderate users (30-99):        SVD=0.4, CF=0.3, CB=0.3
# Light users (<30 ratings):     SVD=0.2, CF=0.2, CB=0.6
#
# Rationale:
# - Active users have rich CF/SVD data — trust latent factors
# - Light users need content-based to compensate for sparse data
# ------------------------------------------------------------

def get_user_weights(user_id):
    """Return (svd_w, cf_w, cb_w) based on user activity."""
    n_ratings = len(ratings[ratings['user_id'] == user_id])
    if n_ratings >= 100:
        return 0.6, 0.3, 0.1   # active user
    elif n_ratings >= 30:
        return 0.4, 0.3, 0.3   # moderate user
    else:
        return 0.2, 0.2, 0.6   # light user


def pred_hybrid(user_id, movie_id):
    """
    Hybrid prediction combining SVD + Item CF + Content-based.
    Weights adapt to user activity level.
    """
    w_svd, w_cf, w_cb = get_user_weights(user_id)

    p_svd = pred_svd(user_id, movie_id)
    p_cf  = pred_item_cf(user_id, movie_id)
    p_cb  = pred_content(user_id, movie_id)

    hybrid = w_svd * p_svd + w_cf * p_cf + w_cb * p_cb
    return float(np.clip(hybrid, 1, 5))


# ------------------------------------------------------------
# STEP 4 — EVALUATE HYBRID MODEL
# ------------------------------------------------------------
print('\nEvaluating hybrid model...')
test_sample = test.sample(500, random_state=42)

preds_h, actuals_h = [], []
for _, row in test_sample.iterrows():
    uid  = int(row['user_id'])
    mid  = int(row['movie_id'])
    true = float(row['rating'])
    if uid in user_item.index and mid in user_item.columns:
        pred = pred_hybrid(uid, mid)
        preds_h.append(pred)
        actuals_h.append(true)

preds_h   = np.array(preds_h)
actuals_h = np.array(actuals_h)

rmse_h = np.sqrt(mean_squared_error(actuals_h, preds_h))
mae_h  = mean_absolute_error(actuals_h, preds_h)

print(f'Hybrid model results (n=500):')
print(f'  RMSE: {rmse_h:.4f}')
print(f'  MAE:  {mae_h:.4f}')

# ------------------------------------------------------------
# STEP 5 — FULL MODEL COMPARISON
# ------------------------------------------------------------
print('\n--- FULL MODEL COMPARISON ---')

# Load individual model results
cf_res  = pd.read_csv('data/processed/cf_results.csv')
svd_res = pd.read_csv('data/processed/svd_results.csv')
cb_res  = pd.read_csv('data/processed/cb_results.csv')

comparison = pd.DataFrame({
    'Model'    : ['Baseline (mean)', 'Content-based',
                  'User-based CF', 'Item-based CF',
                  f'SVD (K={K})', 'Hybrid'],
    'RMSE'     : [1.1116, cb_res['rmse'].values[0],
                  cf_res.loc[0, 'rmse'],
                  cf_res.loc[1, 'rmse'],
                  svd_res['rmse'].values[0], rmse_h],
    'MAE'      : [None, cb_res['mae'].values[0],
                  cf_res.loc[0, 'mae'],
                  cf_res.loc[1, 'mae'],
                  svd_res['mae'].values[0], mae_h]
})

comparison['RMSE_improvement'] = (
    (1.1116 - comparison['RMSE']) / 1.1116 * 100
).round(1)

print(comparison.to_string(index=False))

# ------------------------------------------------------------
# STEP 6 — TOP-10 HYBRID RECOMMENDATIONS
# Generate for multiple users to demonstrate variety
# ------------------------------------------------------------

def get_hybrid_recommendations(user_id, n=10):
    """Generate Top-N hybrid recommendations for a user."""
    rated      = set(user_item.loc[user_id][
        user_item.loc[user_id] > 0].index)
    well_rated = set(
        movies[movies['rating_count'] >= 20]['movie_id'])
    candidates = [m for m in user_item.columns
                  if m not in rated and m in well_rated]

    scores = [(mid, pred_hybrid(user_id, mid))
              for mid in candidates]
    scores.sort(key=lambda x: x[1], reverse=True)

    results = []
    for movie_id, score in scores[:n]:
        title = movies.loc[
            movies['movie_id'] == movie_id, 'title'].values
        title = title[0] if len(title) > 0 \
            else f'Movie {movie_id}'
        results.append({
            'movie_id'        : movie_id,
            'title'           : title,
            'predicted_rating': round(score, 2)
        })
    return pd.DataFrame(results)


# Demonstrate for 3 different users
for uid in [1, 50, 200]:
    n_rated = len(ratings[ratings['user_id'] == uid])
    w = get_user_weights(uid)
    print(f'\n--- HYBRID RECOMMENDATIONS: USER {uid} ---')
    print(f'Ratings given: {n_rated} | '
          f'Weights SVD={w[0]}, CF={w[1]}, CB={w[2]}\n')
    recs = get_hybrid_recommendations(uid, n=10)
    print(recs.to_string(index=False))

# ------------------------------------------------------------
# STEP 7 — VISUALISATIONS
# ------------------------------------------------------------

# Chart 1 — model comparison bar chart
fig, ax = plt.subplots(figsize=(11, 6))
models = comparison['Model'].tolist()
rmses  = comparison['RMSE'].tolist()
colors = [GRAY, CORAL, BLUE_LIGHT, BLUE_MID,
          TEAL, BLUE_DARK]

bars = ax.bar(models, rmses, color=colors,
              width=0.6, zorder=3)
ax.axhline(y=1.1116, color=GRAY, linewidth=1.5,
           linestyle='--', label='Baseline RMSE')

for bar, val in zip(bars, rmses):
    if val:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f'{val:.3f}', ha='center',
                fontsize=9, fontweight='bold',
                color=BLUE_DARK)

ax.set_ylabel('RMSE (lower is better)')
ax.set_title('Model comparison — RMSE on test set\n'
             'SVD and hybrid significantly outperform baseline',
             fontweight='bold')
ax.set_xticklabels(models, rotation=15, ha='right')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('outputs/charts/py14_model_comparison.png',
            dpi=150, bbox_inches='tight')
plt.close()
print('\nChart saved: py14_model_comparison.png')

# Chart 2 — adaptive weights diagram
fig, ax = plt.subplots(figsize=(10, 5))
user_types  = ['Light user\n(<30 ratings)',
               'Moderate user\n(30-99)',
               'Active user\n(100+ ratings)']
svd_weights = [0.2, 0.4, 0.6]
cf_weights  = [0.2, 0.3, 0.3]
cb_weights  = [0.6, 0.3, 0.1]
x = np.arange(len(user_types))
w = 0.25

ax.bar(x - w, svd_weights, w, label='SVD',
       color=BLUE_DARK, zorder=3)
ax.bar(x,     cf_weights,  w, label='Item-based CF',
       color=TEAL, zorder=3)
ax.bar(x + w, cb_weights,  w, label='Content-based',
       color=CORAL, zorder=3)

ax.set_xticks(x)
ax.set_xticklabels(user_types)
ax.set_ylabel('Model weight')
ax.set_title('Adaptive hybrid weights by user activity\n'
             'Content-based dominates for light users, '
             'SVD for active users',
             fontweight='bold')
ax.legend(fontsize=10)
ax.set_ylim(0, 0.75)
plt.tight_layout()
plt.savefig('outputs/charts/py15_adaptive_weights.png',
            dpi=150, bbox_inches='tight')
plt.close()
print('Chart saved: py15_adaptive_weights.png')

# Chart 3 — prediction error comparison
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.hist(preds_h - actuals_h, bins=25,
        color=BLUE_DARK, edgecolor='white',
        linewidth=0.8, zorder=3)
ax.axvline(x=0, color=CORAL, linewidth=2,
           linestyle='--')
ax.set_xlabel('Prediction error')
ax.set_ylabel('Count')
ax.set_title(f'Hybrid model prediction errors\n'
             f'RMSE={rmse_h:.3f}, MAE={mae_h:.3f}',
             fontweight='bold')

ax2 = axes[1]
improvement = comparison['RMSE_improvement'].fillna(0)
colors_imp  = [TEAL if v > 0 else CORAL for v in improvement]
ax2.barh(comparison['Model'], improvement,
         color=colors_imp, height=0.6, zorder=3)
ax2.axvline(x=0, color='black', linewidth=1)
ax2.set_xlabel('RMSE improvement over baseline (%)')
ax2.set_title('Model improvement vs baseline\n'
              'Higher = better',
              fontweight='bold')
for i, val in enumerate(improvement):
    ax2.text(val + 0.3, i,
             f'{val:.1f}%' if val else '',
             va='center', fontsize=9,
             color=BLUE_DARK)

plt.tight_layout()
plt.savefig('outputs/charts/py16_hybrid_evaluation.png',
            dpi=150, bbox_inches='tight')
plt.close()
print('Chart saved: py16_hybrid_evaluation.png')

# ------------------------------------------------------------
# STEP 8 — SAVE FINAL RESULTS
# ------------------------------------------------------------
comparison.to_csv('data/processed/model_comparison.csv',
                  index=False)

hybrid_results = pd.DataFrame({
    'model'    : ['Hybrid (SVD + CF + Content)'],
    'rmse'     : [rmse_h],
    'mae'      : [mae_h],
    'test_size': [len(preds_h)]
})
hybrid_results.to_csv('data/processed/hybrid_results.csv',
                      index=False)

print('\n--- FINAL MODEL SUMMARY ---')
print(comparison[['Model', 'RMSE',
                  'RMSE_improvement']].to_string(index=False))
print('\nScript 06 complete.')
print('Phase 4 — Machine Learning complete.')
