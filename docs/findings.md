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

## Finding 5 — War and Drama rate highest, Horror rates lowest

**What the data shows:**
War films have the highest average rating at 3.82 followed by Drama
at 3.69 and Documentary at 3.67. Horror has the lowest average
rating at 3.29. Comedy also rates below average at 3.39 despite
being the second most common genre.

**What it means:**
Genre is a strong signal for expected rating. War and Documentary
films attract viewers with genuine interest in the subject matter,
producing higher and more consistent ratings. Horror and Comedy
attract casual viewers with more variable reactions.

**ML implication:**
Genre preference is a key feature for content-based filtering.
A user who consistently rates War films above 4.0 is very different
from one who rates Horror films above 4.0. The genre vector for
each user will be weighted by their rating history per genre,
not just by which genres they have watched.

## Finding 6 — age predicts rating generosity, gender does not

**What the data shows:**
Male and female users rate identically at 3.53 average. However age
shows a clear pattern — users under 18 average 3.42 while users 55+
average 3.65. Students make up the largest occupation group at 196
users. Educators rate highest at 3.67 among occupations.

**What it means:**
Gender is not a useful feature for rating prediction but age is.
The 0.23 point gap between youngest and oldest users represents a
meaningful difference in rating behaviour. A personalised model must
account for each user's individual rating baseline rather than
assuming all users use the scale the same way.

**ML implication:**
User average rating will be used as a normalisation factor in the
collaborative filtering model. Rather than predicting raw ratings,
the model will predict deviations from each user's personal average.
This technique — called mean-centering — significantly improves
recommendation accuracy for datasets with user rating bias.
