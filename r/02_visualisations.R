# ============================================================
#  FILMIQ — R SCRIPT 2: VISUALISATIONS
#  File    : r/02_visualisations.R
#  Purpose : Additional publication-quality charts for the
#            FilmIQ portfolio — recommendation diversity,
#            user taste profiles, model comparison dashboard
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
# STEP 2 — LOAD DATA
# ------------------------------------------------------------
PROJECT_DIR <- "C:/Users/User/Documents/filmiq-movie-recommendation"
setwd(PROJECT_DIR)

ratings          <- read_csv("data/processed/ratings.csv")
movies           <- read_csv("data/processed/movies.csv")
model_comparison <- read_csv("data/processed/model_comparison.csv")
user_affinity    <- read_csv("data/processed/user_affinity.csv")

# Recommendation files
recs_usercf <- read_csv("data/processed/user1_recs_usercf.csv")
recs_itemcf <- read_csv("data/processed/user1_recs_itemcf.csv")
recs_svd    <- read_csv("data/processed/user1_recs_svd.csv")
recs_cb     <- read_csv("data/processed/user1_recs_cb.csv")

cat("Data loaded.\n")

BLUE_DARK  <- "#0C447C"
BLUE_MID   <- "#378ADD"
BLUE_LIGHT <- "#B5D4F4"
TEAL       <- "#1D9E75"
CORAL      <- "#D85A30"
AMBER      <- "#BA7517"
GRAY       <- "#888780"

# ------------------------------------------------------------
# CHART 1 — RECOMMENDATION OVERLAP BETWEEN MODELS
# How different are the recommendations from each model?
# ------------------------------------------------------------
cat("Building recommendation overlap analysis...\n")

# Get top-10 movie IDs from each model
top_usercf <- head(recs_usercf$movie_id, 10)
top_itemcf <- head(recs_itemcf$movie_id, 10)
top_svd    <- head(recs_svd$movie_id, 10)
top_cb     <- head(recs_cb$movie_id, 10)

# Calculate pairwise overlap
overlap_matrix <- matrix(0, nrow = 4, ncol = 4,
  dimnames = list(
    c("User CF", "Item CF", "SVD", "Content"),
    c("User CF", "Item CF", "SVD", "Content")
  )
)

lists <- list(top_usercf, top_itemcf, top_svd, top_cb)
for (i in 1:4) {
  for (j in 1:4) {
    overlap_matrix[i, j] <- length(
      intersect(lists[[i]], lists[[j]]))
  }
}

cat("\nRecommendation overlap (shared movies in Top-10):\n")
print(overlap_matrix)

overlap_df <- as.data.frame(overlap_matrix) %>%
  rownames_to_column("Model1") %>%
  pivot_longer(-Model1, names_to = "Model2",
               values_to = "shared_movies")

p1 <- overlap_df %>%
  mutate(
    Model1 = factor(Model1,
      levels = c("User CF", "Item CF", "SVD", "Content")),
    Model2 = factor(Model2,
      levels = c("User CF", "Item CF", "SVD", "Content"))
  ) %>%
  ggplot(aes(x = Model2, y = Model1,
             fill = shared_movies)) +
  geom_tile(colour = "white", linewidth = 1) +
  geom_text(aes(label = shared_movies),
            fontface = "bold", size = 5,
            colour = ifelse(overlap_df$shared_movies > 5,
                           "white", BLUE_DARK)) +
  scale_fill_gradient(low = BLUE_LIGHT, high = BLUE_DARK,
                      name = "Shared\nmovies") +
  labs(
    title    = "Recommendation overlap between models — User 1",
    subtitle = "Number of shared movies in Top-10 recommendations",
    x        = NULL, y = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title    = element_text(face = "bold", size = 13),
    plot.subtitle = element_text(colour = "grey50"),
    axis.text     = element_text(size = 11)
  )

ggsave("outputs/charts/r06_recommendation_overlap.png",
       p1, width = 8, height = 6, dpi = 150)
cat("Chart saved: r06_recommendation_overlap.png\n")

# ------------------------------------------------------------
# CHART 2 — TOP MOVIES BY RATING COUNT AND AVERAGE
# Bubble chart — size = rating count, position = avg rating
# ------------------------------------------------------------
cat("Building popularity vs quality bubble chart...\n")

movie_summary <- ratings %>%
  group_by(movie_id) %>%
  summarise(
    rating_count = n(),
    avg_rating   = mean(rating),
    .groups      = "drop"
  ) %>%
  filter(rating_count >= 50) %>%
  inner_join(movies %>% select(movie_id, title),
             by = "movie_id") %>%
  mutate(title_short = ifelse(
    nchar(title) > 20,
    paste0(substr(title, 1, 20), "..."),
    title
  ))

# Top movies to label
top_label <- movie_summary %>%
  filter(avg_rating >= 4.2 | rating_count >= 400)

p2 <- movie_summary %>%
  ggplot(aes(x = rating_count, y = avg_rating,
             size = rating_count)) +
  geom_point(alpha = 0.5, colour = BLUE_MID) +
  geom_point(data = top_label,
             aes(x = rating_count, y = avg_rating),
             colour = CORAL, size = 3, alpha = 0.8) +
  geom_text(data = top_label,
            aes(label = title_short),
            size = 2.8, vjust = -1,
            colour = BLUE_DARK) +
  geom_hline(yintercept = mean(ratings$rating),
             colour = GRAY, linewidth = 0.8,
             linetype = "dashed") +
  scale_size_continuous(range = c(2, 10),
                        guide = "none") +
  scale_x_continuous(labels = comma) +
  labs(
    title    = "Movie popularity vs quality",
    subtitle = paste0("Movies with 50+ ratings | ",
                      "Red = highlighted films | ",
                      "Dashed = global mean"),
    x        = "Number of ratings (popularity)",
    y        = "Average rating (quality)"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title    = element_text(face = "bold", size = 13),
    plot.subtitle = element_text(colour = "grey50", size = 9)
  )

ggsave("outputs/charts/r07_popularity_vs_quality.png",
       p2, width = 11, height = 7, dpi = 150)
cat("Chart saved: r07_popularity_vs_quality.png\n")

# ------------------------------------------------------------
# CHART 3 — USER TASTE PROFILE RADAR CHART
# Show User 1's genre affinity vs average user
# ------------------------------------------------------------
cat("Building user taste profile...\n")

genre_affinity_cols <- c(
  "action_avg", "adventure_avg", "animation_avg",
  "comedy_avg", "crime_avg", "drama_avg",
  "horror_avg", "romance_avg", "scifi_affinity",
  "thriller_avg", "war_avg"
)

# Check which columns exist
available_cols <- intersect(genre_affinity_cols,
                            colnames(user_affinity))

if (length(available_cols) >= 5 &&
    1 %in% user_affinity[[1]]) {

  user1_affinity <- user_affinity %>%
    filter(row_number() == 1) %>%
    select(all_of(available_cols))

  avg_affinity <- user_affinity %>%
    select(all_of(available_cols)) %>%
    summarise(across(everything(), mean, na.rm = TRUE))

  genre_names <- gsub("_avg|_affinity", "",
                      available_cols)
  genre_names <- tools::toTitleCase(genre_names)

  profile_df <- data.frame(
    genre   = rep(genre_names, 2),
    affinity = c(as.numeric(user1_affinity),
                 as.numeric(avg_affinity)),
    user    = c(rep("User 1", length(genre_names)),
                rep("Average user",
                    length(genre_names)))
  )

  p3 <- profile_df %>%
    ggplot(aes(x = reorder(genre, affinity),
               y = affinity, fill = user)) +
    geom_col(position = "dodge", width = 0.7) +
    geom_hline(yintercept = mean(ratings$rating),
               colour = GRAY, linewidth = 0.8,
               linetype = "dashed") +
    scale_fill_manual(values = c(
      "User 1"       = BLUE_DARK,
      "Average user" = BLUE_LIGHT)) +
    coord_flip() +
    labs(
      title    = "User 1 genre affinity vs average user",
      subtitle = "Average rating given to movies of each genre",
      x        = NULL,
      y        = "Average rating given",
      fill     = NULL
    ) +
    theme_minimal(base_size = 11) +
    theme(
      plot.title      = element_text(face = "bold", size = 13),
      plot.subtitle   = element_text(colour = "grey50"),
      legend.position = "top"
    )

  ggsave("outputs/charts/r08_user_taste_profile.png",
         p3, width = 10, height = 6, dpi = 150)
  cat("Chart saved: r08_user_taste_profile.png\n")
} else {
  cat("Skipping taste profile — column names differ\n")
}

# ------------------------------------------------------------
# CHART 4 — RATING VOLUME OVER TIME
# When were most ratings given?
# ------------------------------------------------------------
cat("Building rating volume over time...\n")

ratings_time <- ratings %>%
  mutate(rated_at = as.POSIXct(rated_at),
         year     = format(rated_at, "%Y"),
         month    = format(rated_at, "%Y-%m")) %>%
  filter(!is.na(rated_at)) %>%
  count(month) %>%
  arrange(month)

if (nrow(ratings_time) > 0) {
  p4 <- ratings_time %>%
    ggplot(aes(x = month, y = n, group = 1)) +
    geom_line(colour = BLUE_DARK, linewidth = 1.5) +
    geom_area(fill = BLUE_LIGHT, alpha = 0.4) +
    scale_y_continuous(labels = comma) +
    labs(
      title    = "Rating volume over time",
      subtitle = "When were the 100,000 ratings given?",
      x        = "Month",
      y        = "Number of ratings"
    ) +
    theme_minimal(base_size = 11) +
    theme(
      plot.title    = element_text(face = "bold", size = 13),
      plot.subtitle = element_text(colour = "grey50"),
      axis.text.x   = element_text(angle = 45, hjust = 1,
                                   size = 8)
    )

  ggsave("outputs/charts/r09_ratings_over_time.png",
         p4, width = 11, height = 6, dpi = 150)
  cat("Chart saved: r09_ratings_over_time.png\n")
}

# ------------------------------------------------------------
# CHART 5 — MODEL PERFORMANCE DASHBOARD
# Combined view of all key metrics
# ------------------------------------------------------------
cat("Building model performance dashboard...\n")

model_clean <- model_comparison %>%
  filter(!is.na(RMSE) & !is.na(MAE)) %>%
  arrange(RMSE) %>%
  mutate(
    Model = factor(Model, levels = Model),
    RMSE_improvement = round(
      (1.1116 - RMSE) / 1.1116 * 100, 1)
  )

p5 <- model_clean %>%
  pivot_longer(cols = c(RMSE, MAE),
               names_to  = "metric",
               values_to = "value") %>%
  ggplot(aes(x = Model, y = value,
             fill = metric)) +
  geom_col(position = "dodge", width = 0.7) +
  geom_text(aes(label = round(value, 3)),
            position = position_dodge(width = 0.7),
            vjust = -0.4, size = 3,
            fontface = "bold") +
  scale_fill_manual(values = c("RMSE" = BLUE_DARK,
                               "MAE"  = TEAL)) +
  scale_y_continuous(limits = c(0, 1.3)) +
  labs(
    title    = "Model performance — RMSE and MAE comparison",
    subtitle = "Lower is better | SVD achieves best accuracy",
    x        = NULL,
    y        = "Error metric value",
    fill     = "Metric"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title      = element_text(face = "bold", size = 13),
    plot.subtitle   = element_text(colour = "grey50"),
    legend.position = "top",
    axis.text.x     = element_text(angle = 15, hjust = 1)
  )

ggsave("outputs/charts/r10_model_dashboard.png",
       p5, width = 11, height = 6, dpi = 150)
cat("Chart saved: r10_model_dashboard.png\n")

cat("\nAll R visualisation charts saved to outputs/charts/\n")
cat("Script 02 complete.\n")
cat("Phase 5 — R analysis complete.\n")
