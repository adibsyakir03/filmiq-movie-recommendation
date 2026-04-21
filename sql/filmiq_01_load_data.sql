-- ============================================================
--  FILMIQ — SCRIPT 1: LOAD DATA
--  File    : sql/01_load_data.sql
--  Purpose : Create schema and load MovieLens 100K dataset
--  Run in  : MySQL Workbench, database filmiq
--  Data    : data/raw/ml-100k/
-- ============================================================

-- ============================================================
-- STEP 1 — CREATE DATABASE
-- ============================================================

CREATE DATABASE IF NOT EXISTS filmiq;
USE filmiq;

-- ============================================================
-- STEP 2 — CREATE TABLES
-- ============================================================

DROP TABLE IF EXISTS ratings;
DROP TABLE IF EXISTS movies;
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS genres;

-- Users table
CREATE TABLE users (
    user_id     INT         NOT NULL,
    age         INT         NULL,
    gender      CHAR(1)     NULL,
    occupation  VARCHAR(50) NULL,
    zip_code    VARCHAR(10) NULL,
    PRIMARY KEY (user_id)
);

-- Movies table
-- Genre columns are binary flags (1 = belongs to genre, 0 = does not)
CREATE TABLE movies (
    movie_id            INT             NOT NULL,
    title               VARCHAR(255)    NOT NULL,
    release_date        VARCHAR(50)     NULL,
    video_release_date  VARCHAR(50)     NULL,
    imdb_url            VARCHAR(255)    NULL,
    unknown_genre       TINYINT         DEFAULT 0,
    action              TINYINT         DEFAULT 0,
    adventure           TINYINT         DEFAULT 0,
    animation           TINYINT         DEFAULT 0,
    children            TINYINT         DEFAULT 0,
    comedy              TINYINT         DEFAULT 0,
    crime               TINYINT         DEFAULT 0,
    documentary         TINYINT         DEFAULT 0,
    drama               TINYINT         DEFAULT 0,
    fantasy             TINYINT         DEFAULT 0,
    film_noir           TINYINT         DEFAULT 0,
    horror              TINYINT         DEFAULT 0,
    musical             TINYINT         DEFAULT 0,
    mystery             TINYINT         DEFAULT 0,
    romance             TINYINT         DEFAULT 0,
    sci_fi              TINYINT         DEFAULT 0,
    thriller            TINYINT         DEFAULT 0,
    war                 TINYINT         DEFAULT 0,
    western             TINYINT         DEFAULT 0,
    PRIMARY KEY (movie_id)
);

-- Ratings table
CREATE TABLE ratings (
    user_id     INT         NOT NULL,
    movie_id    INT         NOT NULL,
    rating      TINYINT     NOT NULL,
    rated_at    DATETIME    NULL,
    PRIMARY KEY (user_id, movie_id),
    FOREIGN KEY (user_id)  REFERENCES users(user_id),
    FOREIGN KEY (movie_id) REFERENCES movies(movie_id)
);

-- Genres lookup table
CREATE TABLE genres (
    genre_id    INT         NOT NULL,
    genre_name  VARCHAR(50) NOT NULL,
    PRIMARY KEY (genre_id)
);

-- ============================================================
-- STEP 3 — LOAD DATA
-- Update paths to match your machine
-- Use forward slashes even on Windows
-- ============================================================

-- Load users
LOAD DATA INFILE 'C:/Users/User/Documents/filmiq-movie-recommendation/data/raw/ml-100k/u.user'
INTO TABLE users
FIELDS TERMINATED BY '|'
LINES TERMINATED BY '\n'
(user_id, age, gender, occupation, zip_code);

-- Load movies
LOAD DATA INFILE 'C:/Users/User/Documents/filmiq-movie-recommendation/data/raw/ml-100k/u.item'
INTO TABLE movies
FIELDS TERMINATED BY '|'
LINES TERMINATED BY '\n'
(
    movie_id, title, release_date, video_release_date, imdb_url,
    unknown_genre, action, adventure, animation, children, comedy,
    crime, documentary, drama, fantasy, film_noir, horror, musical,
    mystery, romance, sci_fi, thriller, war, western
);

-- Load ratings (u.data is tab-separated, timestamp needs converting)
LOAD DATA INFILE 'C:/Users/User/Documents/filmiq-movie-recommendation/data/raw/ml-100k/u.data'
INTO TABLE ratings
FIELDS TERMINATED BY '\t'
LINES TERMINATED BY '\n'
(user_id, movie_id, rating, @ts)
SET rated_at = FROM_UNIXTIME(@ts);

-- Load genres
LOAD DATA INFILE 'C:/Users/User/Documents/filmiq-movie-recommendation/data/raw/ml-100k/u.genre'
INTO TABLE genres
FIELDS TERMINATED BY '|'
LINES TERMINATED BY '\n'
(genre_name, genre_id);

-- ============================================================
-- STEP 4 — VALIDATE THE LOAD
-- ============================================================

-- Row counts
SELECT 'users'   AS table_name, COUNT(*) AS rows FROM users
UNION ALL
SELECT 'movies',  COUNT(*) FROM movies
UNION ALL
SELECT 'ratings', COUNT(*) FROM ratings
UNION ALL
SELECT 'genres',  COUNT(*) FROM genres;

-- Expected:
--   users   = 943
--   movies  = 1682
--   ratings = 100000
--   genres  = 19

-- Rating distribution — should be 1 to 5
SELECT rating, COUNT(*) AS count,
       ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) AS pct
FROM ratings
GROUP BY rating
ORDER BY rating;

-- Top 10 most rated movies
SELECT m.title,
       COUNT(r.rating)       AS rating_count,
       ROUND(AVG(r.rating), 2) AS avg_rating
FROM ratings r
JOIN movies m ON r.movie_id = m.movie_id
GROUP BY m.movie_id, m.title
ORDER BY rating_count DESC
LIMIT 10;

-- Users with most ratings
SELECT user_id,
       COUNT(*)              AS ratings_given,
       ROUND(AVG(rating), 2) AS avg_rating
FROM ratings
GROUP BY user_id
ORDER BY ratings_given DESC
LIMIT 10;

-- Matrix sparsity check
SELECT
    COUNT(DISTINCT user_id)                         AS total_users,
    COUNT(DISTINCT movie_id)                        AS total_movies,
    COUNT(*)                                        AS total_ratings,
    COUNT(DISTINCT user_id) * COUNT(DISTINCT movie_id) AS possible_ratings,
    ROUND(COUNT(*) * 100.0 /
        (COUNT(DISTINCT user_id) *
         COUNT(DISTINCT movie_id)), 2)              AS density_pct
FROM ratings;

-- NOTE: expect ~6.3% density — the matrix is very sparse
-- This is normal for recommendation systems and is the
-- core challenge collaborative filtering must overcome
