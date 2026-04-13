# Data dictionary — MovieLens 100K

**Source:** GroupLens Research, University of Minnesota  
**Citation:** F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets:
History and Context. ACM Transactions on Interactive Intelligent Systems, 5(4):19.  
**Download:** https://grouplens.org/datasets/movielens/100k/

---

## Files overview

| File | Rows | Description |
|---|---|---|
| `u.data` | 100,000 | All ratings — the core dataset |
| `u.item` | 1,682 | Movie metadata |
| `u.user` | 943 | User demographics |
| `u.genre` | 19 | Genre list |
| `u1.base` / `u1.test` | 80K / 20K | Pre-split train/test sets |

---

## u.data — ratings

Tab-separated. No header row.

| Field | Type | Description |
|---|---|---|
| `user_id` | INT | Unique user identifier (1–943) |
| `movie_id` | INT | Unique movie identifier (1–1682) |
| `rating` | INT | Rating given (1–5 stars) |
| `timestamp` | INT | Unix timestamp of the rating |

**Notes:**
- Every user has rated at least 20 movies
- The matrix is sparse — 100,000 ratings out of a possible 943 × 1,682 = 1,586,126 combinations = 6.3% density

---

## u.item — movies

Pipe-separated. No header row.

| Field | Type | Description |
|---|---|---|
| `movie_id` | INT | Unique movie identifier |
| `title` | VARCHAR | Movie title including release year |
| `release_date` | VARCHAR | Release date (DD-Mon-YYYY) |
| `video_release_date` | VARCHAR | Video release date (mostly empty) |
| `imdb_url` | VARCHAR | URL to IMDB page |
| `unknown` | TINYINT | Genre flag (1 = yes, 0 = no) |
| `Action` | TINYINT | Genre flag |
| `Adventure` | TINYINT | Genre flag |
| `Animation` | TINYINT | Genre flag |
| `Children` | TINYINT | Genre flag |
| `Comedy` | TINYINT | Genre flag |
| `Crime` | TINYINT | Genre flag |
| `Documentary` | TINYINT | Genre flag |
| `Drama` | TINYINT | Genre flag |
| `Fantasy` | TINYINT | Genre flag |
| `Film-Noir` | TINYINT | Genre flag |
| `Horror` | TINYINT | Genre flag |
| `Musical` | TINYINT | Genre flag |
| `Mystery` | TINYINT | Genre flag |
| `Romance` | TINYINT | Genre flag |
| `Sci-Fi` | TINYINT | Genre flag |
| `Thriller` | TINYINT | Genre flag |
| `War` | TINYINT | Genre flag |
| `Western` | TINYINT | Genre flag |

**Notes:**
- Movies can belong to multiple genres — the genre columns are binary flags not a single category
- Release years range from 1922 to 1998
- Some titles have missing release dates

---

## u.user — users

Pipe-separated. No header row.

| Field | Type | Description |
|---|---|---|
| `user_id` | INT | Unique user identifier |
| `age` | INT | User age |
| `gender` | CHAR(1) | M or F |
| `occupation` | VARCHAR | User occupation |
| `zip_code` | VARCHAR | US zip code |

**Notes:**
- 943 users total
- Age ranges from 7 to 73
- 21 distinct occupations including student, engineer, programmer, educator

---

## u.genre — genres

Pipe-separated.

| Field | Description |
|---|---|
| `genre_name` | Name of the genre |
| `genre_id` | Integer ID (0–18) |

19 genres total.

---

## Key derived fields (calculated in Python / SQL)

| Field | Formula | Meaning |
|---|---|---|
| `avg_rating` | AVG(rating) per movie | Average rating a movie received |
| `rating_count` | COUNT(rating) per movie | How many users rated the movie |
| `user_avg_rating` | AVG(rating) per user | Each user's average rating — their baseline |
| `genre_vector` | Binary array of 19 genres | Movie represented as a genre feature vector |
| `matrix_density` | ratings / (users × movies) | Sparsity of the user-item matrix |

---

## Analytical assumptions

### Train/test split
All models use the pre-defined `u1.base` (80%) and `u1.test` (20%) split
provided by MovieLens. This ensures fair comparison across models.

### Rating threshold for relevance
For precision@K and recall@K calculations, a movie is considered
**relevant** if the predicted or actual rating is ≥ 4.0 (out of 5).
This is consistent with the MovieLens research literature.

### Cold start
Users or movies with fewer than 5 ratings are excluded from collaborative
filtering evaluation. Content-based filtering handles cold start cases
since it does not require rating history.

---

## References

- Harper, F.M. and Konstan, J.A. (2015). The MovieLens Datasets: History
  and Context. ACM Transactions on Interactive Intelligent Systems, 5(4):19.
- Koren, Y., Bell, R. and Volinsky, C. (2009). Matrix Factorization Techniques
  for Recommender Systems. IEEE Computer, 42(8):30-37. (Netflix Prize paper)
- Ricci, F., Rokach, L. and Shapira, B. (2015). Recommender Systems Handbook.
  Springer. (Standard reference for collaborative and content-based filtering)
