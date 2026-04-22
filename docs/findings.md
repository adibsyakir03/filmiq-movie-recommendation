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
