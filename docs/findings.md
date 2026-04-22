# Findings & analysis commentary

## Project overview

FilmIQ builds a personalised movie recommendation engine using the
MovieLens 100K dataset. Three models are compared — collaborative
filtering, SVD matrix factorisation, and content-based filtering —
then combined into a hybrid model.

Findings will be added progressively as each phase is completed.

---

## Finding 1 — positivity bias in ratings

**What the data shows:**
34.2% of all ratings are 4 stars and 21.2% are 5 stars. Only 17.5%
of ratings are below 3. The most common rating is 4 out of 5.

**What it means:**
Users exhibit positivity bias — they rate movies they chose to watch,
and they chose them because they expected to like them. This skews
the rating distribution toward higher values and makes it harder for
models to distinguish between a good and a great movie.

**ML implication:**
Predicted ratings will cluster around 3.5–4.0. RMSE will be lower
than if ratings were uniformly distributed. Models should be evaluated
on ranking quality (precision@K) not just rating prediction accuracy.

---

## Finding 2 — 47% of users have sparse rating history

**What the data shows:**
21.1% of users are light users with fewer than 30 ratings and 26.2%
are moderate users with 30–59 ratings. Only 15.8% are power users
with 200+ ratings.

**What it means:**
Nearly half the user base has insufficient rating history for
collaborative filtering to work reliably. The hybrid model is
essential — content-based filtering handles sparse users while
collaborative filtering handles active users.

**ML implication:**
The cold start problem affects 21% of users directly. Minimum
rating threshold of 20 ratings will be applied for collaborative
filtering evaluation, excluding light users from CF metrics.

## Finding 3 — quality vs controversy: two types of highly rated films

**What the data shows:**
The top rated films (B1) show low standard deviations (0.73–0.83)
confirming broad consensus. The most controversial films (B2) show
standard deviations above 1.30 — nearly double — with average ratings
clustering around 3.0 indicating deeply split opinions.

**What it means:**
Two distinct recommendation risk profiles exist. Consensus films like
Schindler's List and Casablanca are safe recommendations for almost
any user. Divisive films like Koyaanisqatsi and Natural Born Killers
require precise user matching — only recommend to users with
demonstrated taste for that specific style.

**ML implication:**
The hybrid model should incorporate rating standard deviation as a
confidence signal. High std films require stronger user-movie similarity
evidence before being recommended. This is implemented in the
content-based filtering layer using genre specificity weighting.

## Finding 4 — no unrated movies, but popularity is highly skewed

**What the data shows:**
Every movie in the dataset has at least one rating — zero cold start
movies. However popularity is heavily skewed — Star Wars has 583
ratings while many movies have fewer than 5. The most popular movie
has 116 times more ratings than the median movie.

**What it means:**
Collaborative filtering will work for all movies but will be more
reliable for popular films with many ratings. Obscure films with
fewer than 20 ratings will have unreliable similarity scores.

**ML implication:**
A minimum rating threshold of 20 will be applied when evaluating
collaborative filtering recommendations. Movies below this threshold
will rely on content-based filtering instead.
