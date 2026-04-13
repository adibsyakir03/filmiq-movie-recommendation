# FilmIQ — Personalised Movie Discovery
### A machine learning recommendation engine built on real MovieLens data

![Status](https://img.shields.io/badge/status-in%20progress-yellow)
![Data](https://img.shields.io/badge/data-MovieLens%20100K-blue)
![Tools](https://img.shields.io/badge/tools-MySQL%20%7C%20Python%20%7C%20R-green)

---

## Business question

Given a user's movie ratings history, which movies should they watch next?
This project builds and compares three recommendation approaches —
collaborative filtering, SVD matrix factorisation, and content-based filtering —
then combines them into a hybrid model evaluated using standard ML metrics.

---

## Dataset

**Source:** MovieLens 100K — GroupLens Research, University of Minnesota  
**Size:** 100,000 ratings, 9,742 movies, 610 users  
**Rating scale:** 1–5 stars  
**Download:** https://grouplens.org/datasets/movielens/100k/

> Raw data is not committed to this repository. Download the dataset from the
> link above and place the `ml-100k` folder in `data/raw/` before running any scripts.

**Key files:**

| File | Description |
|---|---|
| `u.data` | 100,000 ratings — user_id, movie_id, rating, timestamp |
| `u.item` | Movie metadata — title, release date, genres |
| `u.user` | User demographics — age, gender, occupation, zip |
| `u.genre` | Genre list |

---

## Methodology

This project implements and compares three recommendation approaches:

1. **Collaborative filtering** — find users similar to you, recommend what they liked
2. **SVD matrix factorisation** — decompose the user-item rating matrix into latent factors (the Netflix Prize method)
3. **Content-based filtering** — recommend movies similar to ones you rated highly, based on genre and tags
4. **Hybrid model** — weighted combination of collaborative + content-based

---

## Models & evaluation metrics

| Metric | What it measures |
|---|---|
| RMSE | Root mean squared error — how far off predicted ratings are |
| MAE | Mean absolute error — average prediction error |
| Precision@K | Of top-K recommendations, what % are relevant |
| Recall@K | Of all relevant movies, what % appear in top-K |

---

## Tools & stack

| Layer | Tool | Purpose |
|---|---|---|
| Database | MySQL 8.0 | Data storage, EDA queries, feature views |
| EDA & ML | Python (`surprise`, `scikit-learn`, `pandas`) | Models, feature engineering, visualisation |
| Evaluation | R (`ggplot2`) | Statistical model comparison, charts |
| Narrative | Jupyter notebook | End-to-end walkthrough |

---

## Repository structure

```
filmiq-movie-recommendation/
│
├── data/
│   ├── raw/ml-100k/          # MovieLens files (not committed)
│   └── processed/            # Feature matrices, model outputs
│
├── sql/
│   ├── 01_load_data.sql      # Schema + data load
│   ├── 02_eda_queries.sql    # Exploratory analysis
│   └── 03_feature_views.sql  # ML feature views
│
├── python/
│   ├── 01_eda.py             # EDA and visualisations
│   ├── 02_features.py        # Feature engineering
│   ├── 03_collaborative_filtering.py
│   ├── 04_svd_model.py
│   ├── 05_content_based.py
│   └── 06_hybrid_model.py
│
├── r/
│   ├── 01_model_evaluation.R
│   └── 02_visualisations.R
│
├── notebooks/
│   └── filmiq_walkthrough.ipynb
│
├── docs/
│   ├── data_dictionary.md
│   └── findings.md
│
├── outputs/
│   └── charts/
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Key findings

> *(To be completed after analysis)*

---

## How to reproduce

### 1. Clone the repo
```bash
git clone https://github.com/adibsyakir03/filmiq-movie-recommendation.git
cd filmiq-movie-recommendation
```

### 2. Download the data
Download `ml-100k.zip` from the GroupLens link above and place the
`ml-100k` folder in `data/raw/`.

### 3. Set up MySQL
```bash
mysql -u root -p < sql/01_load_data.sql
```

### 4. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 5. Run in order
```bash
python python/01_eda.py
python python/02_features.py
python python/03_collaborative_filtering.py
python python/04_svd_model.py
python python/05_content_based.py
python python/06_hybrid_model.py
```

---

## Author

**Adib Syakir**  
Aspiring Data Scientist  
[LinkedIn](https://www.linkedin.com/in/adib-syakir-b05605336)

---

*Data source: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets:
History and Context. ACM Transactions on Interactive Intelligent Systems, 5(4):19.*
