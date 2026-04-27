-- ============================================================
--  FILMIQ — SCRIPT 3: FEATURE VIEWS
--  File    : sql/03_feature_views.sql
--  Purpose : Create views that feed directly into the Python
--            ML pipeline — user profiles, movie stats,
--            genre affinity, user-item matrix export
--  Run in  : MySQL Workbench, database filmiq
-- ============================================================

USE filmiq;

-- ============================================================
-- VIEW 1 — MOVIE STATS
-- Aggregated movie-level features for content-based filtering
-- Used by: Python 02_features.py, 05_content_based.py
-- ============================================================

CREATE OR REPLACE VIEW v_movie_stats AS
SELECT
    m.movie_id,
    m.title,
    m.release_date,
    -- Genre flags
    m.action, m.adventure, m.animation, m.children,
    m.comedy, m.crime, m.documentary, m.drama,
    m.fantasy, m.film_noir, m.horror, m.musical,
    m.mystery, m.romance, m.sci_fi, m.thriller,
    m.war, m.western,
    -- Rating statistics
    COUNT(r.rating)                                     AS rating_count,
    ROUND(AVG(r.rating), 4)                             AS avg_rating,
    ROUND(STD(r.rating), 4)                             AS rating_std,
    MIN(r.rating)                                       AS min_rating,
    MAX(r.rating)                                       AS max_rating,
    -- Popularity score — log scale to reduce dominance of blockbusters
    ROUND(LOG(1 + COUNT(r.rating)), 4)                  AS popularity_score,
    -- Bayesian average rating — shrinks toward global mean for movies
    -- with few ratings. More reliable than raw average.
    -- Formula: (C * m + n * R) / (C + n)
    -- where C = min ratings threshold (50), m = global mean, R = movie mean
    ROUND(
        (50 * 3.53 + COUNT(r.rating) * AVG(r.rating))
        / (50 + COUNT(r.rating)), 4
    )                                                   AS bayesian_avg,
    -- Number of genres this movie belongs to
    (m.action + m.adventure + m.animation + m.children +
     m.comedy + m.crime + m.documentary + m.drama +
     m.fantasy + m.film_noir + m.horror + m.musical +
     m.mystery + m.romance + m.sci_fi + m.thriller +
     m.war + m.western)                                 AS genre_count
FROM movies m
LEFT JOIN ratings r ON m.movie_id = r.movie_id
GROUP BY
    m.movie_id, m.title, m.release_date,
    m.action, m.adventure, m.animation, m.children,
    m.comedy, m.crime, m.documentary, m.drama,
    m.fantasy, m.film_noir, m.horror, m.musical,
    m.mystery, m.romance, m.sci_fi, m.thriller,
    m.war, m.western;


-- ============================================================
-- VIEW 2 — USER PROFILES
-- Aggregated user-level features including genre affinity
-- Used by: Python 02_features.py, 03_collaborative_filtering.py
-- ============================================================

CREATE OR REPLACE VIEW v_user_profiles AS
SELECT
    u.user_id,
    u.age,
    u.gender,
    u.occupation,
    -- Rating behaviour
    COUNT(r.rating)                                     AS movies_rated,
    ROUND(AVG(r.rating), 4)                             AS avg_rating,
    ROUND(STD(r.rating), 4)                             AS rating_std,
    -- User type classification
    CASE
        WHEN COUNT(r.rating) < 30  THEN 'light'
        WHEN COUNT(r.rating) < 100 THEN 'moderate'
        ELSE 'active'
    END                                                 AS user_type,
    -- Genre affinity scores
    -- Average rating given to movies of each genre
    -- NULL if user has not rated any movie in that genre
    ROUND(AVG(CASE WHEN m.action    = 1 THEN r.rating END), 3) AS action_affinity,
    ROUND(AVG(CASE WHEN m.adventure = 1 THEN r.rating END), 3) AS adventure_affinity,
    ROUND(AVG(CASE WHEN m.animation = 1 THEN r.rating END), 3) AS animation_affinity,
    ROUND(AVG(CASE WHEN m.comedy    = 1 THEN r.rating END), 3) AS comedy_affinity,
    ROUND(AVG(CASE WHEN m.crime     = 1 THEN r.rating END), 3) AS crime_affinity,
    ROUND(AVG(CASE WHEN m.drama     = 1 THEN r.rating END), 3) AS drama_affinity,
    ROUND(AVG(CASE WHEN m.horror    = 1 THEN r.rating END), 3) AS horror_affinity,
    ROUND(AVG(CASE WHEN m.romance   = 1 THEN r.rating END), 3) AS romance_affinity,
    ROUND(AVG(CASE WHEN m.sci_fi    = 1 THEN r.rating END), 3) AS scifi_affinity,
    ROUND(AVG(CASE WHEN m.thriller  = 1 THEN r.rating END), 3) AS thriller_affinity,
    ROUND(AVG(CASE WHEN m.war       = 1 THEN r.rating END), 3) AS war_affinity
FROM users u
LEFT JOIN ratings r  ON u.user_id  = r.user_id
LEFT JOIN movies  m  ON r.movie_id = m.movie_id
GROUP BY u.user_id, u.age, u.gender, u.occupation;


-- ============================================================
-- VIEW 3 — NORMALISED RATINGS
-- Mean-centered ratings for collaborative filtering
-- Removes user rating bias (generous vs harsh raters)
-- Used by: Python 03_collaborative_filtering.py, 04_svd_model.py
-- ============================================================

CREATE OR REPLACE VIEW v_normalised_ratings AS
SELECT
    r.user_id,
    r.movie_id,
    r.rating                                            AS raw_rating,
    up.avg_rating                                       AS user_mean,
    ROUND(r.rating - up.avg_rating, 4)                  AS normalised_rating,
    r.rated_at
FROM ratings r
JOIN v_user_profiles up ON r.user_id = up.user_id;

-- Why normalise?
-- User A always rates 4-5. User B always rates 1-3.
-- Both give Movie X a 4. But User B's 4 is much stronger praise.
-- Normalising subtracts each user's mean so ratings are comparable.


-- ============================================================
-- VIEW 4 — TRAIN SET (80%) AND TEST SET (20%)
-- Uses MovieLens pre-defined split for fair model comparison
-- Based on u1.base and u1.test files
-- Used by: Python ML scripts for model evaluation
-- ============================================================

-- NOTE: The actual train/test split will be loaded from
-- u1.base and u1.test files in Python using the Surprise library.
-- This view provides a random 80/20 split as an alternative.

CREATE OR REPLACE VIEW v_train_ratings AS
SELECT
    user_id,
    movie_id,
    rating,
    rated_at
FROM ratings
WHERE MOD(user_id * 1000 + movie_id, 5) != 0;  -- 80% approx

CREATE OR REPLACE VIEW v_test_ratings AS
SELECT
    user_id,
    movie_id,
    rating,
    rated_at
FROM ratings
WHERE MOD(user_id * 1000 + movie_id, 5) = 0;   -- 20% approx


-- ============================================================
-- VIEW 5 — TOP MOVIES PER GENRE
-- Best movie in each genre by Bayesian average rating
-- Used by: Python 05_content_based.py for cold start
-- ============================================================

CREATE OR REPLACE VIEW v_top_movies_by_genre AS
SELECT genre, movie_id, title, bayesian_avg, rating_count
FROM (
    SELECT 'Action'      AS genre, movie_id, title,
           bayesian_avg, rating_count,
           ROW_NUMBER() OVER (ORDER BY bayesian_avg DESC) AS rn
    FROM v_movie_stats WHERE action = 1 AND rating_count >= 20
    UNION ALL
    SELECT 'Adventure',  movie_id, title, bayesian_avg, rating_count,
           ROW_NUMBER() OVER (ORDER BY bayesian_avg DESC)
    FROM v_movie_stats WHERE adventure = 1 AND rating_count >= 20
    UNION ALL
    SELECT 'Comedy',     movie_id, title, bayesian_avg, rating_count,
           ROW_NUMBER() OVER (ORDER BY bayesian_avg DESC)
    FROM v_movie_stats WHERE comedy = 1 AND rating_count >= 20
    UNION ALL
    SELECT 'Drama',      movie_id, title, bayesian_avg, rating_count,
           ROW_NUMBER() OVER (ORDER BY bayesian_avg DESC)
    FROM v_movie_stats WHERE drama = 1 AND rating_count >= 20
    UNION ALL
    SELECT 'Horror',     movie_id, title, bayesian_avg, rating_count,
           ROW_NUMBER() OVER (ORDER BY bayesian_avg DESC)
    FROM v_movie_stats WHERE horror = 1 AND rating_count >= 20
    UNION ALL
    SELECT 'Sci-Fi',     movie_id, title, bayesian_avg, rating_count,
           ROW_NUMBER() OVER (ORDER BY bayesian_avg DESC)
    FROM v_movie_stats WHERE sci_fi = 1 AND rating_count >= 20
    UNION ALL
    SELECT 'Thriller',   movie_id, title, bayesian_avg, rating_count,
           ROW_NUMBER() OVER (ORDER BY bayesian_avg DESC)
    FROM v_movie_stats WHERE thriller = 1 AND rating_count >= 20
    UNION ALL
    SELECT 'Romance',    movie_id, title, bayesian_avg, rating_count,
           ROW_NUMBER() OVER (ORDER BY bayesian_avg DESC)
    FROM v_movie_stats WHERE romance = 1 AND rating_count >= 20
    UNION ALL
    SELECT 'War',        movie_id, title, bayesian_avg, rating_count,
           ROW_NUMBER() OVER (ORDER BY bayesian_avg DESC)
    FROM v_movie_stats WHERE war = 1 AND rating_count >= 20
    UNION ALL
    SELECT 'Animation',  movie_id, title, bayesian_avg, rating_count,
           ROW_NUMBER() OVER (ORDER BY bayesian_avg DESC)
    FROM v_movie_stats WHERE animation = 1 AND rating_count >= 20
) AS genre_ranked
WHERE rn <= 10
ORDER BY genre, rn;


-- ============================================================
-- VERIFY ALL VIEWS
-- ============================================================

SHOW FULL TABLES IN filmiq WHERE TABLE_TYPE = 'VIEW';

-- Quick preview
SELECT COUNT(*) AS movie_stats_rows    FROM v_movie_stats;
SELECT COUNT(*) AS user_profile_rows   FROM v_user_profiles;
SELECT COUNT(*) AS normalised_rows     FROM v_normalised_ratings;
SELECT COUNT(*) AS train_rows          FROM v_train_ratings;
SELECT COUNT(*) AS test_rows           FROM v_test_ratings;

-- Sample user profile — user 1
SELECT * FROM v_user_profiles WHERE user_id = 1;

-- Sample movie stats — Star Wars
SELECT movie_id, title, avg_rating, bayesian_avg,
       rating_count, popularity_score
FROM v_movie_stats
WHERE title LIKE '%Star Wars%';
