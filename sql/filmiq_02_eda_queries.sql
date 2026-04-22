-- ============================================================
--  FILMIQ — SCRIPT 2: EXPLORATORY DATA ANALYSIS
--  File    : sql/02_eda_queries.sql
--  Purpose : Explore the MovieLens 100K dataset
--            Understand rating patterns, user behaviour,
--            movie popularity and genre distributions
--  Run in  : MySQL Workbench, database filmiq
-- ============================================================

USE filmiq;

-- ============================================================
-- SECTION A — RATING ANALYSIS
-- Understand how users rate movies
-- ============================================================

-- A1. Overall rating distribution
-- What is the shape of ratings across all users?
SELECT
    rating,
    COUNT(*)                                            AS count,
    ROUND(COUNT(*) * 100.0
        / SUM(COUNT(*)) OVER(), 1)                      AS pct,
    REPEAT('█', COUNT(*) / 1000)                        AS bar
FROM ratings
GROUP BY rating
ORDER BY rating;


-- A2. Average rating per user
-- Do some users rate more generously than others?
SELECT
    user_id,
    COUNT(*)                    AS movies_rated,
    ROUND(AVG(rating), 2)       AS avg_rating,
    MIN(rating)                 AS min_rating,
    MAX(rating)                 AS max_rating,
    ROUND(STD(rating), 2)       AS rating_std
FROM ratings
GROUP BY user_id
ORDER BY movies_rated DESC
LIMIT 20;


-- A3. Rating activity over time
-- When were most ratings given?
SELECT
    YEAR(rated_at)              AS year,
    MONTH(rated_at)             AS month,
    COUNT(*)                    AS ratings_given
FROM ratings
WHERE rated_at IS NOT NULL
GROUP BY YEAR(rated_at), MONTH(rated_at)
ORDER BY year, month;


-- A4. User rating behaviour buckets
-- How many movies does the average user rate?
SELECT
    CASE
        WHEN movies_rated < 30  THEN 'Light user (< 30)'
        WHEN movies_rated < 60  THEN 'Moderate user (30-59)'
        WHEN movies_rated < 100 THEN 'Active user (60-99)'
        WHEN movies_rated < 200 THEN 'Heavy user (100-199)'
        ELSE 'Power user (200+)'
    END                         AS user_type,
    COUNT(*)                    AS user_count,
    ROUND(COUNT(*) * 100.0
        / SUM(COUNT(*)) OVER(), 1) AS pct
FROM (
    SELECT user_id, COUNT(*) AS movies_rated
    FROM ratings
    GROUP BY user_id
) AS user_activity
GROUP BY user_type
ORDER BY MIN(movies_rated);


-- ============================================================
-- SECTION B — MOVIE ANALYSIS
-- Understand movie popularity and quality
-- ============================================================

-- B1. Top 20 movies by average rating
-- Minimum 50 ratings to filter out obscure films
SELECT
    m.title,
    COUNT(r.rating)             AS rating_count,
    ROUND(AVG(r.rating), 2)     AS avg_rating,
    ROUND(STD(r.rating), 2)     AS rating_std
FROM ratings r
JOIN movies m ON r.movie_id = m.movie_id
GROUP BY m.movie_id, m.title
HAVING rating_count >= 50
ORDER BY avg_rating DESC
LIMIT 20;


-- B2. Most controversial movies
-- High rating count + high standard deviation = divisive film
SELECT
    m.title,
    COUNT(r.rating)             AS rating_count,
    ROUND(AVG(r.rating), 2)     AS avg_rating,
    ROUND(STD(r.rating), 2)     AS rating_std
FROM ratings r
JOIN movies m ON r.movie_id = m.movie_id
GROUP BY m.movie_id, m.title
HAVING rating_count >= 50
ORDER BY rating_std DESC
LIMIT 15;


-- B3. Most popular movies by rating count
SELECT
    m.title,
    COUNT(r.rating)             AS rating_count,
    ROUND(AVG(r.rating), 2)     AS avg_rating
FROM ratings r
JOIN movies m ON r.movie_id = m.movie_id
GROUP BY m.movie_id, m.title
ORDER BY rating_count DESC
LIMIT 20;


-- B4. Movies with no ratings — cold start problem
-- These cannot be recommended by collaborative filtering
SELECT
    m.movie_id,
    m.title
FROM movies m
LEFT JOIN ratings r ON m.movie_id = r.movie_id
WHERE r.movie_id IS NULL
ORDER BY m.movie_id
LIMIT 20;

SELECT COUNT(*) AS unrated_movies
FROM movies m
LEFT JOIN ratings r ON m.movie_id = r.movie_id
WHERE r.movie_id IS NULL;


-- ============================================================
-- SECTION C — GENRE ANALYSIS
-- Which genres are most popular?
-- ============================================================

-- C1. Number of movies per genre
SELECT
    g.genre_name,
    SUM(CASE g.genre_name
        WHEN 'Action'      THEN m.action
        WHEN 'Adventure'   THEN m.adventure
        WHEN 'Animation'   THEN m.animation
        WHEN 'Children'    THEN m.children
        WHEN 'Comedy'      THEN m.comedy
        WHEN 'Crime'       THEN m.crime
        WHEN 'Documentary' THEN m.documentary
        WHEN 'Drama'       THEN m.drama
        WHEN 'Fantasy'     THEN m.fantasy
        WHEN 'Film-Noir'   THEN m.film_noir
        WHEN 'Horror'      THEN m.horror
        WHEN 'Musical'     THEN m.musical
        WHEN 'Mystery'     THEN m.mystery
        WHEN 'Romance'     THEN m.romance
        WHEN 'Sci-Fi'      THEN m.sci_fi
        WHEN 'Thriller'    THEN m.thriller
        WHEN 'War'         THEN m.war
        WHEN 'Western'     THEN m.western
        ELSE 0
    END)                        AS movie_count
FROM genres g
CROSS JOIN movies m
WHERE g.genre_name != 'unknown'
GROUP BY g.genre_name
ORDER BY movie_count DESC;


-- C2. Average rating by genre
-- Which genres do users rate highest?
SELECT
    genre_name,
    movie_count,
    ROUND(avg_rating, 2)        AS avg_rating,
    total_ratings
FROM (
    SELECT 'Drama'       AS genre_name,
           SUM(m.drama)  AS movie_count,
           AVG(CASE WHEN m.drama = 1 THEN r.rating END) AS avg_rating,
           SUM(CASE WHEN m.drama = 1 THEN 1 ELSE 0 END) AS total_ratings
    FROM movies m JOIN ratings r ON m.movie_id = r.movie_id
    UNION ALL
    SELECT 'Comedy', SUM(m.comedy),
           AVG(CASE WHEN m.comedy = 1 THEN r.rating END),
           SUM(CASE WHEN m.comedy = 1 THEN 1 ELSE 0 END)
    FROM movies m JOIN ratings r ON m.movie_id = r.movie_id
    UNION ALL
    SELECT 'Action', SUM(m.action),
           AVG(CASE WHEN m.action = 1 THEN r.rating END),
           SUM(CASE WHEN m.action = 1 THEN 1 ELSE 0 END)
    FROM movies m JOIN ratings r ON m.movie_id = r.movie_id
    UNION ALL
    SELECT 'Thriller', SUM(m.thriller),
           AVG(CASE WHEN m.thriller = 1 THEN r.rating END),
           SUM(CASE WHEN m.thriller = 1 THEN 1 ELSE 0 END)
    FROM movies m JOIN ratings r ON m.movie_id = r.movie_id
    UNION ALL
    SELECT 'Romance', SUM(m.romance),
           AVG(CASE WHEN m.romance = 1 THEN r.rating END),
           SUM(CASE WHEN m.romance = 1 THEN 1 ELSE 0 END)
    FROM movies m JOIN ratings r ON m.movie_id = r.movie_id
    UNION ALL
    SELECT 'Horror', SUM(m.horror),
           AVG(CASE WHEN m.horror = 1 THEN r.rating END),
           SUM(CASE WHEN m.horror = 1 THEN 1 ELSE 0 END)
    FROM movies m JOIN ratings r ON m.movie_id = r.movie_id
    UNION ALL
    SELECT 'Sci-Fi', SUM(m.sci_fi),
           AVG(CASE WHEN m.sci_fi = 1 THEN r.rating END),
           SUM(CASE WHEN m.sci_fi = 1 THEN 1 ELSE 0 END)
    FROM movies m JOIN ratings r ON m.movie_id = r.movie_id
    UNION ALL
    SELECT 'War', SUM(m.war),
           AVG(CASE WHEN m.war = 1 THEN r.rating END),
           SUM(CASE WHEN m.war = 1 THEN 1 ELSE 0 END)
    FROM movies m JOIN ratings r ON m.movie_id = r.movie_id
    UNION ALL
    SELECT 'Documentary', SUM(m.documentary),
           AVG(CASE WHEN m.documentary = 1 THEN r.rating END),
           SUM(CASE WHEN m.documentary = 1 THEN 1 ELSE 0 END)
    FROM movies m JOIN ratings r ON m.movie_id = r.movie_id
    UNION ALL
    SELECT 'Animation', SUM(m.animation),
           AVG(CASE WHEN m.animation = 1 THEN r.rating END),
           SUM(CASE WHEN m.animation = 1 THEN 1 ELSE 0 END)
    FROM movies m JOIN ratings r ON m.movie_id = r.movie_id
) AS genre_stats
ORDER BY avg_rating DESC;


-- ============================================================
-- SECTION D — USER DEMOGRAPHICS
-- How do different user groups rate movies?
-- ============================================================

-- D1. Average rating by gender
SELECT
    u.gender,
    COUNT(DISTINCT u.user_id)   AS user_count,
    COUNT(r.rating)             AS total_ratings,
    ROUND(AVG(r.rating), 2)     AS avg_rating
FROM users u
JOIN ratings r ON u.user_id = r.user_id
GROUP BY u.gender;


-- D2. Average rating by age group
SELECT
    CASE
        WHEN u.age < 18  THEN 'Under 18'
        WHEN u.age < 25  THEN '18-24'
        WHEN u.age < 35  THEN '25-34'
        WHEN u.age < 45  THEN '35-44'
        WHEN u.age < 55  THEN '45-54'
        ELSE '55+'
    END                         AS age_group,
    COUNT(DISTINCT u.user_id)   AS user_count,
    COUNT(r.rating)             AS total_ratings,
    ROUND(AVG(r.rating), 2)     AS avg_rating
FROM users u
JOIN ratings r ON u.user_id = r.user_id
GROUP BY age_group
ORDER BY MIN(u.age);


-- D3. Top occupations by number of ratings
SELECT
    u.occupation,
    COUNT(DISTINCT u.user_id)   AS user_count,
    COUNT(r.rating)             AS total_ratings,
    ROUND(AVG(r.rating), 2)     AS avg_rating
FROM users u
JOIN ratings r ON u.user_id = r.user_id
GROUP BY u.occupation
ORDER BY total_ratings DESC;


-- ============================================================
-- SECTION E — RECOMMENDATION READINESS
-- Assess data quality for ML pipeline
-- ============================================================

-- E1. User-movie matrix density by user activity
-- Active users have denser rows — better for collaborative filtering
SELECT
    CASE
        WHEN rating_count < 30  THEN 'Light (< 30 ratings)'
        WHEN rating_count < 100 THEN 'Moderate (30-99)'
        ELSE 'Active (100+)'
    END                         AS user_segment,
    COUNT(*)                    AS users,
    ROUND(AVG(rating_count), 0) AS avg_ratings_per_user,
    ROUND(AVG(avg_rating), 2)   AS avg_rating
FROM (
    SELECT
        user_id,
        COUNT(*)            AS rating_count,
        AVG(rating)         AS avg_rating
    FROM ratings
    GROUP BY user_id
) AS user_stats
GROUP BY user_segment
ORDER BY MIN(rating_count);


-- E2. Movie coverage — what % of movies have enough ratings for CF?
SELECT
    CASE
        WHEN rating_count = 0   THEN 'No ratings (cold start)'
        WHEN rating_count < 5   THEN 'Very few (1-4)'
        WHEN rating_count < 20  THEN 'Low (5-19)'
        WHEN rating_count < 50  THEN 'Moderate (20-49)'
        ELSE 'Well rated (50+)'
    END                         AS coverage_tier,
    COUNT(*)                    AS movie_count,
    ROUND(COUNT(*) * 100.0
        / SUM(COUNT(*)) OVER(), 1) AS pct
FROM (
    SELECT
        m.movie_id,
        COUNT(r.rating)     AS rating_count
    FROM movies m
    LEFT JOIN ratings r ON m.movie_id = r.movie_id
    GROUP BY m.movie_id
) AS movie_coverage
GROUP BY coverage_tier
ORDER BY MIN(rating_count);
