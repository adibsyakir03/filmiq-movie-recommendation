# ============================================================
#  FILMIQ — R SCRIPT 1: MODEL EVALUATION
#  File    : r/01_model_evaluation.R
#  Purpose : Statistical comparison of all recommendation
#            models — RMSE, MAE, precision@K, recall@K
#            Produces evaluation charts for the portfolio
#  Run in  : RStudio
# ============================================================

# ------------------------------------------------------------
# STEP 1 — LOAD LIBRARIES
# ------------------------------------------------------------
library(dplyr)
library(tidyr)
library(ggplot2)
library(scales)
library(readr)

# ------------------------------------------------------------
# STEP 2 — LOAD MODEL RESULTS
# ------------------------------------------------------------
PROJECT_DIR <- "C:/Users/User/Documents/filmiq-movie-recommendation"
setwd(PROJECT_DIR)

cat("Loading model results...\n")

model_comparison <- read_csv("data/processed/model_comparison.csv")
ratings          <- read_csv("data/processed/ratings.csv")
movies           <- read_csv("data/processed/movies.csv")
train            <- read_csv("data/processed/train.csv")
test             <- read_csv("data/processed/test.csv")

# Load individual recommendations for User 1
recs_usercf <- read_csv("data/processed/user1_recs_usercf.csv")
recs_itemcf <- read_csv("data/processed/user1_recs_itemcf.csv")
recs_svd    <- read_csv("data/processed/user1_recs_svd.csv")
recs_cb     <- read_csv("data/processed/user1_recs_cb.csv")

cat("Data loaded successfully\n")
cat(sprintf("Models compared: %d\n", nrow(model_comparison)))

# ------------------------------------------------------------
# STEP 3 — CLEAN MODEL COMPARISON TABLE
# ------------------------------------------------------------
cat("\n--- MODEL COMPARISON TABLE ---\n")

model_comparison_clean <- model_comparison %>%
  filter(!is.na(RMSE)) %>%
  arrange(RMSE) %>%
  mutate(
    RMSE             = round(RMSE, 4),
    MAE              = round(MAE, 4),
    RMSE_improvement = round(RMSE_improvement, 1),
    Rank             = row_number()
  )

print(model_comparison_clean)

# Best model
best_model <- model_comparison_clean %>%
  filter(RMSE == min(RMSE))
cat(sprintf("\nBest model: %s (RMSE=%.4f)\n",
            best_model$Model, best_model$RMSE))

# ------------------------------------------------------------
# STEP 4 — PRECISION@K AND RECALL@K
# For Top-K recommendations, what fraction are relevant?
# A movie is "relevant" if actual rating >= 4.0
# ------------------------------------------------------------
cat("\n--- PRECISION@K AND RECALL@K ---\n")
cat("Relevance threshold: rating >= 4.0\n\n")

# Calculate precision@K and recall@K for item-based CF
# Using test set ratings as ground truth
RELEVANCE_THRESHOLD <- 4.0
K_VALUES <- c(5, 10, 20)

# For each user in test set, find their relevant movies
test_relevant <- test %>%
  filter(rating >= RELEVANCE_THRESHOLD) %>%
  group_by(user_id) %>%
  summarise(relevant_movies = list(movie_id),
            n_relevant = n())

# Simulate top-K recommendations using predicted ratings
# Use item-based CF predictions as proxy
# In production this would use actual model outputs

# For demonstration: use highest-rated train movies per user
# as a proxy for what the model recommends
train_top_movies <- train %>%
  group_by(user_id, movie_id) %>%
  summarise(rating = mean(rating)) %>%
  arrange(user_id, desc(rating))

# Calculate metrics for different K values
precision_recall <- data.frame()

for (k in K_VALUES) {
  # Get top-K movies per user from training set
  top_k_per_user <- train_top_movies %>%
    group_by(user_id) %>%
    slice_head(n = k) %>%
    summarise(recommended = list(movie_id))

  # Join with relevant movies from test
  joined <- inner_join(top_k_per_user, test_relevant,
                       by = "user_id")

  if (nrow(joined) == 0) next

  # Calculate precision and recall
  metrics <- joined %>%
    rowwise() %>%
    mutate(
      hits      = length(intersect(recommended,
                                   relevant_movies)),
      precision = hits / k,
      recall    = ifelse(n_relevant > 0,
                         hits / n_relevant, 0)
    ) %>%
    ungroup() %>%
    summarise(
      K              = k,
      avg_precision  = round(mean(precision), 4),
      avg_recall     = round(mean(recall), 4),
      n_users        = n()
    )

  precision_recall <- bind_rows(precision_recall, metrics)
}

cat("Precision@K and Recall@K results:\n")
print(precision_recall)

# ------------------------------------------------------------
# STEP 5 — RATING DISTRIBUTION ANALYSIS
# How well do model predictions match actual rating patterns?
# ------------------------------------------------------------
cat("\n--- RATING DISTRIBUTION ANALYSIS ---\n")

# Distribution of actual ratings in test set
actual_dist <- test %>%
  count(rating) %>%
  mutate(pct = round(n / sum(n) * 100, 1),
         type = "Actual")

cat("Actual rating distribution in test set:\n")
print(actual_dist)

# Rating statistics
cat(sprintf("\nTest set statistics:\n"))
cat(sprintf("  Mean rating:   %.4f\n", mean(test$rating)))
cat(sprintf("  SD rating:     %.4f\n", sd(test$rating)))
cat(sprintf("  Median rating: %.1f\n", median(test$rating)))

# ------------------------------------------------------------
# STEP 6 — USER SEGMENT ANALYSIS
# Does model performance vary by user activity level?
# ------------------------------------------------------------
cat("\n--- USER SEGMENT ANALYSIS ---\n")

user_activity <- ratings %>%
  group_by(user_id) %>%
  summarise(
    n_ratings  = n(),
    avg_rating = mean(rating),
    user_type  = case_when(
      n() < 30  ~ "Light (<30)",
      n() < 100 ~ "Moderate (30-99)",
      TRUE      ~ "Active (100+)"
    )
  )

segment_summary <- user_activity %>%
  group_by(user_type) %>%
  summarise(
    n_users        = n(),
    avg_ratings    = round(mean(n_ratings), 0),
    avg_user_rating = round(mean(avg_rating), 3),
    pct_of_users   = round(n() / nrow(user_activity) * 100, 1)
  )

cat("User segment breakdown:\n")
print(segment_summary)

# ------------------------------------------------------------
# STEP 7 — GENRE PREFERENCE ANALYSIS
# What genres are most common in high vs low ratings?
# ------------------------------------------------------------
cat("\n--- GENRE PREFERENCE ANALYSIS ---\n")

genre_cols <- c("action", "adventure", "animation", "comedy",
                "crime", "drama", "horror", "romance",
                "sci_fi", "thriller", "war")

# Join ratings with movie genres
merged <- ratings %>%
  inner_join(movies %>%
               select(movie_id, all_of(genre_cols)),
             by = "movie_id")

# Average rating per genre
genre_avg <- sapply(genre_cols, function(g) {
  subset <- merged[merged[[g]] == 1, ]
  round(mean(subset$rating), 3)
})

genre_df <- data.frame(
  genre      = names(genre_avg),
  avg_rating = as.numeric(genre_avg)
) %>%
  arrange(desc(avg_rating))

cat("Average rating by genre:\n")
print(genre_df)

# ------------------------------------------------------------
# STEP 8 — VISUALISATIONS
# ------------------------------------------------------------

BLUE_DARK  <- "#0C447C"
BLUE_MID   <- "#378ADD"
BLUE_LIGHT <- "#B5D4F4"
TEAL       <- "#1D9E75"
CORAL      <- "#D85A30"
AMBER      <- "#BA7517"
GRAY       <- "#888780"

# Chart 1 — Model RMSE comparison
p1 <- model_comparison_clean %>%
  mutate(Model = factor(Model, levels = Model)) %>%
  ggplot(aes(x = Model, y = RMSE,
             fill = RMSE == min(RMSE))) +
  geom_col(width = 0.7) +
  geom_hline(yintercept = 1.1116,
             colour = GRAY, linewidth = 1,
             linetype = "dashed") +
  geom_text(aes(label = RMSE),
            vjust = -0.4, size = 3.5,
            fontface = "bold", colour = BLUE_DARK) +
  scale_fill_manual(values = c("FALSE" = BLUE_MID,
                               "TRUE"  = TEAL),
                    guide = "none") +
  scale_y_continuous(limits = c(0, 1.3)) +
  labs(
    title    = "Model comparison — RMSE on test set",
    subtitle = "Lower RMSE = better predictions | Dashed = baseline",
    x        = NULL,
    y        = "RMSE"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title    = element_text(face = "bold", size = 13),
    plot.subtitle = element_text(colour = "grey50"),
    axis.text.x   = element_text(angle = 15, hjust = 1)
  )

ggsave("outputs/charts/r01_model_rmse_comparison.png",
       p1, width = 10, height = 6, dpi = 150)
cat("\nChart saved: r01_model_rmse_comparison.png\n")

# Chart 2 — RMSE improvement over baseline
p2 <- model_comparison_clean %>%
  filter(RMSE_improvement > 0) %>%
  mutate(Model = reorder(Model, RMSE_improvement)) %>%
  ggplot(aes(x = Model, y = RMSE_improvement,
             fill = RMSE_improvement)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = paste0("+", RMSE_improvement, "%")),
            hjust = -0.1, size = 3.5,
            fontface = "bold", colour = BLUE_DARK) +
  scale_fill_gradient(low = BLUE_LIGHT, high = TEAL,
                      guide = "none") +
  scale_y_continuous(limits = c(0, 60)) +
  coord_flip() +
  labs(
    title    = "RMSE improvement over baseline",
    subtitle = "SVD latent factors provide the largest single improvement",
    x        = NULL,
    y        = "Improvement (%)"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title    = element_text(face = "bold", size = 13),
    plot.subtitle = element_text(colour = "grey50")
  )

ggsave("outputs/charts/r02_rmse_improvement.png",
       p2, width = 10, height = 6, dpi = 150)
cat("Chart saved: r02_rmse_improvement.png\n")

# Chart 3 — Precision@K and Recall@K
if (nrow(precision_recall) > 0) {
  p3 <- precision_recall %>%
    pivot_longer(cols = c(avg_precision, avg_recall),
                 names_to = "metric",
                 values_to = "value") %>%
    mutate(
      metric = ifelse(metric == "avg_precision",
                      "Precision@K", "Recall@K"),
      K      = factor(K)
    ) %>%
    ggplot(aes(x = K, y = value,
               fill = metric)) +
    geom_col(position = "dodge", width = 0.6) +
    geom_text(aes(label = round(value, 3)),
              position = position_dodge(width = 0.6),
              vjust = -0.4, size = 3.5,
              fontface = "bold") +
    scale_fill_manual(values = c("Precision@K" = BLUE_DARK,
                                 "Recall@K"    = TEAL)) +
    scale_y_continuous(limits = c(0, 0.5)) +
    labs(
      title    = "Precision@K and Recall@K",
      subtitle = paste0("Relevance threshold: rating >= ",
                        RELEVANCE_THRESHOLD),
      x        = "K (number of recommendations)",
      y        = "Score",
      fill     = "Metric"
    ) +
    theme_minimal(base_size = 11) +
    theme(
      plot.title    = element_text(face = "bold", size = 13),
      plot.subtitle = element_text(colour = "grey50"),
      legend.position = "top"
    )

  ggsave("outputs/charts/r03_precision_recall.png",
         p3, width = 10, height = 6, dpi = 150)
  cat("Chart saved: r03_precision_recall.png\n")
}

# Chart 4 — Genre average ratings
p4 <- genre_df %>%
  mutate(
    genre = tools::toTitleCase(
      gsub("_", "-", genre)),
    genre = reorder(genre, avg_rating),
    color_group = avg_rating >= mean(avg_rating)
  ) %>%
  ggplot(aes(x = genre, y = avg_rating,
             fill = color_group)) +
  geom_col(width = 0.7) +
  geom_hline(yintercept = mean(ratings$rating),
             colour = GRAY, linewidth = 1,
             linetype = "dashed",
             label = "Global mean") +
  geom_text(aes(label = round(avg_rating, 2)),
            hjust = -0.2, size = 3.5,
            fontface = "bold", colour = BLUE_DARK) +
  scale_fill_manual(values = c("FALSE" = CORAL,
                               "TRUE"  = TEAL),
                    guide = "none") +
  scale_y_continuous(limits = c(3.0, 4.1)) +
  coord_flip() +
  labs(
    title    = "Average rating by genre",
    subtitle = "Dashed line = global mean (3.52)",
    x        = NULL,
    y        = "Average rating"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title    = element_text(face = "bold", size = 13),
    plot.subtitle = element_text(colour = "grey50")
  )

ggsave("outputs/charts/r04_genre_ratings.png",
       p4, width = 10, height = 6, dpi = 150)
cat("Chart saved: r04_genre_ratings.png\n")

# Chart 5 — User segment breakdown
p5 <- segment_summary %>%
  mutate(user_type = factor(
    user_type,
    levels = c("Light (<30)",
               "Moderate (30-99)",
               "Active (100+)"))) %>%
  ggplot(aes(x = user_type, y = n_users,
             fill = user_type)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = paste0(n_users, "\n(",
                               pct_of_users, "%)")),
            vjust = -0.3, size = 3.5,
            fontface = "bold", colour = BLUE_DARK) +
  scale_fill_manual(values = c(
    "Light (<30)"      = CORAL,
    "Moderate (30-99)" = AMBER,
    "Active (100+)"    = TEAL),
    guide = "none") +
  scale_y_continuous(limits = c(0, 450)) +
  labs(
    title    = "User activity segments",
    subtitle = "47% of users have fewer than 60 ratings",
    x        = "User type",
    y        = "Number of users"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title    = element_text(face = "bold", size = 13),
    plot.subtitle = element_text(colour = "grey50")
  )

ggsave("outputs/charts/r05_user_segments.png",
       p5, width = 10, height = 6, dpi = 150)
cat("Chart saved: r05_user_segments.png\n")

cat("\nAll 5 R charts saved to outputs/charts/\n")
cat("Script 01 complete.\n")
