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

## Finding 5 — Film Noir rates highest despite being the rarest genre

**What the data shows:**
Film Noir has the highest average rating at 3.92 despite having only
24 movies in the dataset. War follows at 3.82, then Drama at 3.69.
Horror rates lowest at 3.29. Comedy rates below average at 3.39
despite being the second most common genre with 505 films.

**What it means:**
Film Noir attracts a small but highly engaged audience of serious
cinephiles who rate consistently and thoughtfully. This is the niche
quality effect — passionate audiences produce higher average ratings
than broad mainstream genres. War films benefit from similar dynamics.
Horror and Comedy attract casual viewers with more variable reactions,
pulling averages down.

**ML implication:**
Genre is a strong predictor of expected rating but must be weighted
by genre frequency. Film Noir should not dominate recommendations
simply because its average rating is highest — it needs to be matched
to users who have demonstrated appreciation for classic or arthouse
cinema. The content-based model handles this through user genre
affinity scores rather than raw genre averages.

**Note on methodology:** SQL and Python calculations produce slightly
different genre averages due to differences in how multi-genre movie
ratings are aggregated. Film Noir ranks first in Python (3.92) and
War ranks first in SQL (3.82). Both methods confirm the same pattern —
niche genres with engaged audiences rate higher than mainstream genres.

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

## Finding 7 — only 36% of movies and 39% of users are CF-ready

**What the data shows:**
35.9% of movies have 50+ ratings suitable for reliable collaborative
filtering. 44.1% of movies have fewer than 20 ratings. On the user
side, 21.1% are light users with fewer than 30 ratings — insufficient
for reliable CF similarity calculation.

**What it means:**
A pure collaborative filtering system would fail to recommend 44% of
the movie catalogue and would produce poor recommendations for 21%
of users. This quantifies exactly why the hybrid model is necessary —
not as a theoretical improvement but as a practical requirement.

**ML implication:**
The hybrid model architecture assigns weights dynamically:
- High CF weight for active users rating well-known movies
- High content-based weight for light users or obscure movies
- Equal blending for moderate cases
This adaptive weighting is the key design decision of the FilmIQ
hybrid model.

## Finding 8 — CF models reveal two distinct recommendation styles

**What the data shows:**
User-based CF (RMSE=1.015, MAE=0.821) recommends universally
acclaimed films — Casablanca, Schindler's List, Rear Window.
Item-based CF (RMSE=0.972, MAE=0.741) recommends niche arthouse
films matching specific viewing patterns — City of Lost Children,
Trainspotting, Chungking Express.

Item-based CF outperforms user-based CF on both RMSE and MAE,
consistent with published recommendation research showing item
similarities are more stable than user similarities.

**What it means:**
The two approaches are complementary rather than competing.
User-based CF captures broad taste alignment while item-based
CF captures specific viewing style. A user who rated Shanghai
Triad highly gets very different but equally valid recommendations
from each method.

**ML implication:**
The hybrid model blends both approaches. For users with many
ratings, item-based CF dominates. For users with few ratings,
user-based CF provides more stable recommendations by leveraging
the collective behaviour of similar users.
