# ============================================================
#  FILMIQ — SCRIPT 1: EDA AND VISUALISATIONS
#  File    : python/01_eda.py
#  Purpose : Load data from MySQL, produce EDA charts,
#            understand the dataset before modelling
#  Run in  : VS Code
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sqlalchemy import create_engine
import os
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# STEP 1 — SETUP
# ------------------------------------------------------------
PROJECT_DIR = r'C:\Users\User\Documents\filmiq-movie-recommendation'
os.chdir(PROJECT_DIR)

# Chart style
plt.rcParams.update({
    'figure.facecolor'  : 'white',
    'axes.facecolor'    : 'white',
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'axes.grid'         : True,
    'grid.alpha'        : 0.3,
    'font.family'       : 'sans-serif',
    'font.size'         : 11
})

BLUE_DARK  = '#0C447C'
BLUE_MID   = '#378ADD'
BLUE_LIGHT = '#B5D4F4'
TEAL       = '#1D9E75'
CORAL      = '#D85A30'
AMBER      = '#BA7517'
GRAY       = '#888780'

# ------------------------------------------------------------
# STEP 2 — CONNECT TO MYSQL AND LOAD DATA
# ------------------------------------------------------------
# Update password if needed
engine = create_engine(
    'mysql+pymysql://root:Actuary123@localhost/filmiq'
)

print('Loading data from MySQL...')

ratings  = pd.read_sql('SELECT * FROM ratings',       engine)
movies   = pd.read_sql('SELECT * FROM v_movie_stats', engine)
users    = pd.read_sql('SELECT * FROM v_user_profiles', engine)
norm_rat = pd.read_sql('SELECT * FROM v_normalised_ratings', engine)

print(f'Ratings:  {len(ratings):,} rows')
print(f'Movies:   {len(movies):,} rows')
print(f'Users:    {len(users):,} rows')

# Save to processed folder for later scripts
ratings.to_csv('data/processed/ratings.csv', index=False)
movies.to_csv('data/processed/movies.csv', index=False)
users.to_csv('data/processed/users.csv', index=False)
norm_rat.to_csv('data/processed/normalised_ratings.csv', index=False)
print('Data saved to data/processed/')

# ------------------------------------------------------------
# STEP 3 — CHART 1: RATING DISTRIBUTION
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: count bar chart
ax = axes[0]
rating_counts = ratings['rating'].value_counts().sort_index()
bars = ax.bar(rating_counts.index, rating_counts.values,
              color=[BLUE_LIGHT, BLUE_MID, BLUE_DARK,
                     TEAL, '#0A6B4E'],
              width=0.7, zorder=3)
for bar, val in zip(bars, rating_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 200,
            f'{val:,}', ha='center', fontsize=9,
            fontweight='bold', color=BLUE_DARK)
ax.set_xlabel('Rating (stars)')
ax.set_ylabel('Number of ratings')
ax.set_title('Rating distribution\n100,000 ratings across 5 levels',
             fontweight='bold')
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xticklabels(['1★', '2★', '3★', '4★', '5★'])

# Right: percentage pie chart
ax2 = axes[1]
pct = rating_counts / rating_counts.sum() * 100
colors_pie = [BLUE_LIGHT, BLUE_MID, BLUE_DARK, TEAL, '#0A6B4E']
wedges, texts, autotexts = ax2.pie(
    pct.values,
    labels=[f'{i}★' for i in range(1, 6)],
    colors=colors_pie,
    autopct='%1.1f%%',
    startangle=90,
    pctdistance=0.75
)
for at in autotexts:
    at.set_fontsize(9)
    at.set_fontweight('bold')
ax2.set_title('Rating distribution (%)\nPositivity bias — 55% rated 4★ or 5★',
              fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/charts/py01_rating_distribution.png',
            dpi=150, bbox_inches='tight')
plt.close()
print('Chart saved: py01_rating_distribution.png')

# ------------------------------------------------------------
# STEP 4 — CHART 2: USER ACTIVITY DISTRIBUTION
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

user_activity = ratings.groupby('user_id')['rating'].count()

# Left: histogram
ax = axes[0]
ax.hist(user_activity, bins=30, color=BLUE_MID,
        edgecolor='white', linewidth=0.8, zorder=3)
ax.axvline(x=user_activity.median(), color=CORAL,
           linewidth=2, linestyle='--',
           label=f'Median ({user_activity.median():.0f} ratings)')
ax.axvline(x=user_activity.mean(), color=TEAL,
           linewidth=2, linestyle='-',
           label=f'Mean ({user_activity.mean():.0f} ratings)')
ax.set_xlabel('Number of movies rated per user')
ax.set_ylabel('Number of users')
ax.set_title('User activity distribution\nWide range from 20 to 737 ratings',
             fontweight='bold')
ax.legend(fontsize=9)

# Right: user type breakdown
ax2 = axes[1]
bins    = [0, 29, 59, 99, 199, 1000]
labels  = ['Light\n<30', 'Moderate\n30-59',
           'Active\n60-99', 'Heavy\n100-199', 'Power\n200+']
user_types = pd.cut(user_activity, bins=bins, labels=labels)
type_counts = user_types.value_counts().reindex(labels)
colors_ut = [CORAL, AMBER, BLUE_LIGHT, BLUE_MID, BLUE_DARK]
bars = ax2.bar(type_counts.index, type_counts.values,
               color=colors_ut, width=0.7, zorder=3)
for bar, val in zip(bars, type_counts.values):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 2,
             f'{val}\n({val/len(user_activity)*100:.0f}%)',
             ha='center', fontsize=9,
             fontweight='bold', color=BLUE_DARK)
ax2.set_xlabel('User type')
ax2.set_ylabel('Number of users')
ax2.set_title('Users by activity level\n47% are light or moderate users',
              fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/charts/py02_user_activity.png',
            dpi=150, bbox_inches='tight')
plt.close()
print('Chart saved: py02_user_activity.png')

# ------------------------------------------------------------
# STEP 5 — CHART 3: MOVIE POPULARITY VS QUALITY
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 7))

# Filter to movies with at least 20 ratings
movies_plot = movies[movies['rating_count'] >= 20].copy()

scatter = ax.scatter(
    movies_plot['rating_count'],
    movies_plot['avg_rating'],
    c=movies_plot['bayesian_avg'],
    cmap='RdYlGn',
    alpha=0.6,
    s=40,
    zorder=3
)

# Annotate top movies
top_movies = movies_plot.nlargest(5, 'bayesian_avg')
for _, row in top_movies.iterrows():
    title_short = row['title'][:25] + '...' \
                  if len(row['title']) > 25 else row['title']
    ax.annotate(title_short,
                xy=(row['rating_count'], row['avg_rating']),
                xytext=(10, 5), textcoords='offset points',
                fontsize=8, color=BLUE_DARK)

plt.colorbar(scatter, ax=ax, label='Bayesian average rating')
ax.set_xlabel('Number of ratings (popularity)')
ax.set_ylabel('Average rating (quality)')
ax.set_title('Movie popularity vs quality\n'
             'Colour = Bayesian adjusted rating',
             fontweight='bold')
ax.axhline(y=3.53, color=GRAY, linewidth=1,
           linestyle='--', label='Global mean (3.53)')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('outputs/charts/py03_popularity_vs_quality.png',
            dpi=150, bbox_inches='tight')
plt.close()
print('Chart saved: py03_popularity_vs_quality.png')

# ------------------------------------------------------------
# STEP 6 — CHART 4: GENRE POPULARITY HEATMAP
# ------------------------------------------------------------
# Build genre rating matrix
genre_cols = ['action', 'adventure', 'animation', 'comedy',
              'crime', 'drama', 'fantasy', 'film_noir',
              'horror', 'musical', 'mystery', 'romance',
              'sci_fi', 'thriller', 'war', 'western']

genre_labels = ['Action', 'Adventure', 'Animation', 'Comedy',
                'Crime', 'Drama', 'Fantasy', 'Film Noir',
                'Horror', 'Musical', 'Mystery', 'Romance',
                'Sci-Fi', 'Thriller', 'War', 'Western']

# Merge ratings with movies
merged = ratings.merge(
    movies[['movie_id'] + genre_cols + ['avg_rating']],
    on='movie_id'
)

# Calculate avg rating per genre
genre_stats = []
for col, label in zip(genre_cols, genre_labels):
    subset = merged[merged[col] == 1]
    genre_stats.append({
        'genre'       : label,
        'movie_count' : movies[movies[col] == 1].shape[0],
        'avg_rating'  : subset['rating'].mean(),
        'total_ratings': len(subset)
    })

genre_df = pd.DataFrame(genre_stats).sort_values(
    'avg_rating', ascending=True)

fig, ax = plt.subplots(figsize=(11, 7))
bars = ax.barh(genre_df['genre'], genre_df['avg_rating'],
               color=[TEAL if v >= 3.6 else
                      BLUE_MID if v >= 3.4 else CORAL
                      for v in genre_df['avg_rating']],
               height=0.7, zorder=3)
ax.axvline(x=3.53, color=GRAY, linewidth=1.5,
           linestyle='--', label='Global mean (3.53)')
for bar, val in zip(bars, genre_df['avg_rating']):
    ax.text(val + 0.01,
            bar.get_y() + bar.get_height()/2,
            f'{val:.2f}', va='center', fontsize=9,
            color=BLUE_DARK)
ax.set_xlabel('Average rating')
ax.set_title('Average rating by genre\n'
             'War and Documentary rate highest, Horror lowest',
             fontweight='bold')
ax.legend(fontsize=9)
ax.set_xlim(3.0, 4.0)

plt.tight_layout()
plt.savefig('outputs/charts/py04_genre_ratings.png',
            dpi=150, bbox_inches='tight')
plt.close()
print('Chart saved: py04_genre_ratings.png')

# ------------------------------------------------------------
# STEP 7 — CHART 5: MATRIX SPARSITY VISUALISATION
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: sparsity overview
ax = axes[0]
total_possible = 943 * 1682
total_rated    = 100000
total_missing  = total_possible - total_rated

ax.bar(['Rated', 'Unrated'],
       [total_rated, total_missing],
       color=[TEAL, CORAL], width=0.5, zorder=3)
ax.set_ylabel('Number of user-movie pairs')
ax.set_title('User-item matrix sparsity\n'
             '93.7% of possible ratings are missing',
             fontweight='bold')
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
for i, (label, val) in enumerate(
        zip(['Rated\n(6.3%)', 'Unrated\n(93.7%)'],
            [total_rated, total_missing])):
    ax.text(i, val + 10000, label,
            ha='center', fontsize=10,
            fontweight='bold', color=BLUE_DARK)

# Right: sample of matrix (first 50 users, 100 movies)
ax2 = axes[1]
sample_users  = sorted(ratings['user_id'].unique())[:50]
sample_movies = sorted(ratings['movie_id'].unique())[:100]
sample        = ratings[
    ratings['user_id'].isin(sample_users) &
    ratings['movie_id'].isin(sample_movies)
].copy()

matrix = sample.pivot_table(
    index='user_id', columns='movie_id',
    values='rating', fill_value=0
)
matrix = matrix.reindex(
    index=sample_users,
    columns=sample_movies,
    fill_value=0
)

ax2.imshow(matrix.values, cmap='Blues', aspect='auto',
           interpolation='none')
ax2.set_xlabel('Movie ID (first 100)')
ax2.set_ylabel('User ID (first 50)')
ax2.set_title('Sample of user-item matrix\n'
              'White = no rating, Blue = rating given',
              fontweight='bold')
ax2.set_xticks([])
ax2.set_yticks([])

plt.tight_layout()
plt.savefig('outputs/charts/py05_matrix_sparsity.png',
            dpi=150, bbox_inches='tight')
plt.close()
print('Chart saved: py05_matrix_sparsity.png')

# ------------------------------------------------------------
# STEP 8 — CHART 6: RATING DEMOGRAPHICS
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: avg rating by age group
age_groups = users.copy()
age_groups['age_group'] = pd.cut(
    age_groups['age'],
    bins=[0, 17, 24, 34, 44, 54, 100],
    labels=['<18', '18-24', '25-34',
            '35-44', '45-54', '55+']
)
age_avg = age_groups.groupby(
    'age_group', observed=True)['avg_rating'].mean()

ax = axes[0]
bars = ax.bar(age_avg.index, age_avg.values,
              color=BLUE_MID, width=0.7, zorder=3)
ax.axhline(y=3.53, color=CORAL, linewidth=1.5,
           linestyle='--', label='Global mean (3.53)')
for bar, val in zip(bars, age_avg.values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.005,
            f'{val:.2f}', ha='center', fontsize=9,
            fontweight='bold', color=BLUE_DARK)
ax.set_xlabel('Age group')
ax.set_ylabel('Average rating given')
ax.set_title('Rating generosity by age\n'
             'Older users rate more generously',
             fontweight='bold')
ax.set_ylim(3.3, 3.8)
ax.legend(fontsize=9)

# Right: gender comparison
ax2 = axes[1]
gender_avg = users.groupby('gender')['avg_rating'].mean()
bars2 = ax2.bar(['Female', 'Male'],
                gender_avg.values,
                color=[CORAL, BLUE_MID],
                width=0.5, zorder=3)
for bar, val in zip(bars2, gender_avg.values):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.002,
             f'{val:.3f}', ha='center', fontsize=11,
             fontweight='bold', color=BLUE_DARK)
ax2.set_ylabel('Average rating given')
ax2.set_title('Rating by gender\n'
              'No meaningful difference (both 3.53)',
              fontweight='bold')
ax2.set_ylim(3.4, 3.7)
ax2.axhline(y=3.53, color=GRAY, linewidth=1.5,
            linestyle='--', label='Global mean')
ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig('outputs/charts/py06_rating_demographics.png',
            dpi=150, bbox_inches='tight')
plt.close()
print('Chart saved: py06_rating_demographics.png')

print('\nAll 6 EDA charts saved to outputs/charts/')
print('Script 01 complete.')
