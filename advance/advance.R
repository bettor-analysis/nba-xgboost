# NBA XGBoost Model for Predicting Team Scores
# Adam Wickwire - Bettor Analysis

# Load necessary libraries
library(tidyverse)
library(hoopR)
library(xgboost)
library(caret)
library(doParallel)  
library(lubridate)
        


#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#

# Load data

# Load historical data
historical <- read_csv("data/historical.csv")

current_season <- load_nba_team_box(seasons = 2025)

# remove rows with Eastern Conf All-Stars , Western Conf All-Stars as current_season
current_season <- current_season %>%
  filter(!team_display_name %in% c("Eastern Conf All-Stars", "Western Conf All-Stars"))

# Save current season data to a CSV file
write_csv(current_season, "data/current_season.csv")

current_season <- read_csv("data/current_season.csv")

# Combine historical and current season data
team_box <- bind_rows(historical, current_season)


#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#



# Select data for analysis
data <- team_box %>%
  select(game_id, team_display_name, team_score:turnovers) %>% 
  select(-team_winner)

# Filter for games with >= 0 points in the paint
data <- data %>% 
  filter(points_in_paint >= 0)

# Create new variables for two-point field goals
data <- data %>% 
  mutate(
    two_point_field_goals_made = field_goals_made - three_point_field_goals_made,
    two_point_field_goals_attempted = field_goals_attempted - three_point_field_goals_attempted,
    two_point_field_goals_percentage = round(two_point_field_goals_made / two_point_field_goals_attempted * 100, 1)
  )

# Calculate effective field goal percentage
data <- data %>% 
  mutate(
    effective_field_goal_percentage = (two_point_field_goals_made + 1.5 * three_point_field_goals_made) / field_goals_attempted
  )

# Calculate true shooting percentage 
data <- data %>% 
  mutate(
    true_shooting_percentage = team_score / (2 * (field_goals_attempted + 0.44 * free_throws_attempted))
  )

# Calculate possessions 
data <- data %>% 
  mutate(
    possessions = 0.5 * (field_goals_attempted + 0.475 * free_throws_attempted - offensive_rebounds + turnovers)
  )

# Calculate Offensive and Defensive Ratings
data <- data %>%
  group_by(game_id) %>%
  mutate(
    opponent_team_score = ifelse(row_number() == 1, lead(team_score), lag(team_score)),
    opponent_possessions = ifelse(row_number() == 1, lead(possessions), lag(possessions)),
    total_possessions = possessions + opponent_possessions,
    OR = 100 * (team_score / total_possessions),
    DR = 100 * (opponent_team_score / total_possessions)
  ) %>%
  ungroup()

# Calculate Free Throw Rate 
data <- data %>% 
  mutate(
    free_throw_rate = free_throws_attempted / field_goals_attempted
  )

# Select data for XGBoost model
model_data <- data %>% 
  select(team_score, effective_field_goal_percentage, true_shooting_percentage, OR, DR)



#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#




# Remove any rows with NA values (optional, based on your data)
model_data <- na.omit(model_data)

# Split the data into training+validation and final test sets
set.seed(123)
final_split <- createDataPartition(model_data$team_score, p = 0.8, list = FALSE)
train_val_data <- model_data[final_split, ]
final_test_data <- model_data[-final_split, ]

# Define predictors and response for training+validation
train_val_predictors <- train_val_data %>% select(-team_score)
train_val_response <- train_val_data$team_score

# Define predictors and response for final test set
test_predictors <- final_test_data %>% select(-team_score)
test_response <- final_test_data$team_score

# Set up cross-validation
set.seed(123)  # For reproducibility
train_control <- trainControl(
  method = "cv",           # Cross-validation
  number = 5,              # 5-fold CV
  verboseIter = TRUE,      # Print progress
  allowParallel = TRUE     # Allow parallel processing
)

# Register parallel backend
cl <- makePSOCKcluster(detectCores() - 1)  # Reserve one core for OS
registerDoParallel(cl)

# Define a reduced grid of hyperparameters
# Focus on key parameters: eta, max_depth, and subsample
xgb_grid <- expand.grid(
  nrounds = 100,            # Number of boosting rounds
  eta = c(0.01, 0.1, 0.3),  # Learning rates
  max_depth = c(4, 6, 8),   # Maximum tree depths
  gamma = 0,                # Minimum loss reduction
  colsample_bytree = 0.8,   # Subsample ratio of columns
  min_child_weight = 1,     # Minimum sum of instance weight
  subsample = 0.8           # Subsample ratio of rows
)

# Alternatively, use a random search with tuneLength
# Uncomment the following block to use tuneLength instead of a predefined grid
# xgb_caret_model <- train(
#   x = train_val_predictors,
#   y = train_val_response,
#   method = "xgbTree",
#   trControl = train_control,
#   tuneLength = 10,          # Number of random combinations to try
#   metric = "RMSE"
# )

# Train the XGBoost model with cross-validation using the reduced grid
xgb_caret_model <- train(
  x = train_val_predictors,
  y = train_val_response,
  method = "xgbTree",
  trControl = train_control,
  tuneGrid = xgb_grid,
  metric = "RMSE"
)

# Stop the parallel cluster
stopCluster(cl)
registerDoSEQ()

# Display the results of cross-validation
# print(xgb_caret_model)
# plot(xgb_caret_model)

# Best model's RMSE from cross-validation
best_rmse <- min(xgb_caret_model$results$RMSE)
# cat("Best CV RMSE: ", best_rmse, "\n")

# Make predictions on the final test set using the best model
final_predictions <- predict(xgb_caret_model, newdata = test_predictors)

# Evaluate model performance on the final test set
final_rmse <- sqrt(mean((final_predictions - test_response)^2))
# cat("Final Test RMSE: ", final_rmse, "\n")

# Feature Importance
importance_matrix <- xgb.importance(feature_names = colnames(train_val_predictors), model = xgb_caret_model$finalModel)
# print(importance_matrix)
# xgb.plot.importance(importance_matrix)




#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#


teams_2025_data <- team_box %>%
  filter(season == 2025)


# Select data for analysis
teams_2025 <- teams_2025_data %>%
  select(game_id, team_display_name, team_score:turnovers) %>% 
  select(-team_winner)

# Filter for games with >= 0 points in the paint
teams_2025 <- teams_2025 %>% 
  filter(points_in_paint >= 0)

# Create new variables for two-point field goals
teams_2025 <- teams_2025 %>% 
  mutate(
    two_point_field_goals_made = field_goals_made - three_point_field_goals_made,
    two_point_field_goals_attempted = field_goals_attempted - three_point_field_goals_attempted,
    two_point_field_goals_percentage = round(two_point_field_goals_made / two_point_field_goals_attempted * 100, 1)
  )

# Calculate effective field goal percentage
teams_2025 <- teams_2025 %>% 
  mutate(
    effective_field_goal_percentage = (two_point_field_goals_made + 1.5 * three_point_field_goals_made) / field_goals_attempted
  )

# Calculate true shooting percentage 
teams_2025 <- teams_2025 %>% 
  mutate(
    true_shooting_percentage = team_score / (2 * (field_goals_attempted + 0.44 * free_throws_attempted))
  )

# Calculate possessions 
teams_2025 <- teams_2025 %>% 
  mutate(
    possessions = 0.5 * (field_goals_attempted + 0.475 * free_throws_attempted - offensive_rebounds + turnovers)
  )

# Calculate Offensive and Defensive Ratings
teams_2025 <- teams_2025 %>%
  group_by(game_id) %>%
  mutate(
    opponent_team_score = ifelse(row_number() == 1, lead(team_score), lag(team_score)),
    opponent_possessions = ifelse(row_number() == 1, lead(possessions), lag(possessions)),
    total_possessions = possessions + opponent_possessions,
    OR = 100 * (team_score / total_possessions),
    DR = 100 * (opponent_team_score / total_possessions)
  ) %>%
  ungroup()

# Calculate Free Throw Rate 
teams_2025 <- teams_2025 %>% 
  mutate(
    free_throw_rate = free_throws_attempted / field_goals_attempted
  )

# Select data for XGBoost model
teams_2025 <- teams_2025 %>% 
  select(team_display_name, team_score, effective_field_goal_percentage, 
         true_shooting_percentage, OR, DR)


# Summarize the data by team with NA handling
teams_2025_summary <- teams_2025 %>%
  group_by(team_display_name) %>%
  summarize(
    team_score = mean(team_score, na.rm = TRUE),
    effective_field_goal_percentage = mean(effective_field_goal_percentage, na.rm = TRUE),
    true_shooting_percentage = mean(true_shooting_percentage, na.rm = TRUE),
    OR = mean(OR, na.rm = TRUE),
    DR = mean(DR, na.rm = TRUE)
  ) %>%
  ungroup()

# Calculate standard deviations for each team score
teams_2025_sd_data <- teams_2025 %>%
  group_by(team_display_name) %>%
  summarize(
    team_score_sd = sd(team_score, na.rm = TRUE),
  ) %>%
  ungroup()


# Make predictions on the 2025 data using the trained model 
teams_2025_predictions <- predict(xgb_caret_model, newdata = teams_2025_summary %>% select(-team_display_name, -team_score))


# Combine predictions with team names
teams_2025_predictions <- teams_2025_summary %>%
  select(team_display_name) %>%
  bind_cols(teams_2025_predictions) 

# rename column called  ...2
colnames(teams_2025_predictions)[2] <- "predicted_score"


# add the standard deviation columns to the predictions. join by team_display_name
teams_2025_predictions <- teams_2025_predictions %>%
  left_join(teams_2025_sd_data, by = "team_display_name")


# simulate the final score for each team by taking the median of the simulation
teams_2025_predictions <- teams_2025_predictions %>% 
  rowwise() %>%
  mutate(
    final_score_n = median(rnorm(10000, mean = predicted_score, sd = team_score_sd))
  )

teams_2025_predictions <- teams_2025_predictions %>% 
  rowwise() %>%
  mutate(
    final_score_p = median(rpois(10000, lambda = predicted_score))
  )


team_score_predictions <- teams_2025_predictions %>%
  select(team_display_name, final_score_p) %>% 
  rename(team = team_display_name, score = final_score_p)


#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#

# get the current seasons schedule 
current_schedule <- load_nba_schedule(seasons = 2025)

# filter the current schedule to only include games that have not been played yet
current_schedule <- current_schedule %>% 
  filter(status_type_name == "STATUS_SCHEDULED")


current_schedule <- current_schedule %>%
  mutate(
    date_only = as.Date(substr(date, 1, 10))
  )


# filter the current schedule for today's games
current_schedule <- current_schedule %>%
  filter(date_only == Sys.Date())

current_schedule <- current_schedule %>%
  select(status_type_detail, home_display_name, away_display_name)

current_schedule <- current_schedule %>% 
  rename(date = status_type_detail, 
         home = home_display_name, 
         away = away_display_name)

# join the team score predictions with the current schedule
current_schedule <- current_schedule %>%
  left_join(team_score_predictions, by = c("home" = "team")) %>%
  rename(home_score = score)

current_schedule <- current_schedule %>%
  left_join(team_score_predictions, by = c("away" = "team")) %>%
  rename(away_score = score)



todays_predictions <- current_schedule %>%
  select(date, home, home_score, away, away_score)
todays_predictions <- todays_predictions %>%
  mutate(
    total_score = home_score + away_score,
    spread = abs(home_score - away_score),
    winner = ifelse(home_score > away_score, home, away)
  )


# calculate win probabilities for each team
# Set a seed for reproducibility
set.seed(123)
# Perform simulation to calculate win probabilities
todays_predictions <- todays_predictions %>%
  rowwise() %>%
  mutate(
    # Simulate 10,000 scores for home and away teams
    home_sim = list(rpois(10000, lambda = home_score)),
    away_sim = list(rpois(10000, lambda = away_score)),
    # Calculate win probabilities
    home_win_prob = mean(unlist(home_sim) > unlist(away_sim)),
    away_win_prob = mean(unlist(away_sim) > unlist(home_sim))
  ) %>%
  ungroup() %>%
  # Remove simulation columns to clean up the dataframe
  select(-home_sim, -away_sim)
# Normalize probabilities to ensure they sum to 1 (optional but recommended)
todays_predictions <- todays_predictions %>%
  mutate(
    total_prob = home_win_prob + away_win_prob,
    home_win_prob = home_win_prob / total_prob,
    away_win_prob = away_win_prob / total_prob
  ) %>%
  select(-total_prob)
# Function to convert win probabilities to American odds
win_prob_to_odds <- function(prob) {
  # Vectorized conversion using ifelse
  odds <- ifelse(
    prob > 0.5,
    round(-(prob / (1 - prob)) * 100),
    ifelse(
      prob < 0.5,
      round((1 - prob) / prob * 100),
      100
    )
  )
  return(odds)
}
# Apply the function to convert win probabilities to American odds
todays_predictions <- todays_predictions %>%
  mutate(
    home_fair_odds = round(win_prob_to_odds(home_win_prob)),
    away_fair_odds = round(win_prob_to_odds(away_win_prob))
  )
todays_predictions <- todays_predictions %>%
  mutate(
    home_win_prob = round(home_win_prob * 100, 2),
    away_win_prob = round(away_win_prob * 100, 2)
  )
# Display the final predictions
# print(todays_predictions)
# write the predictions to a CSV file and date with systems date to can
# track the predictions daily put them in the daily_predictions folder
write_csv(todays_predictions, paste0("daily_predictions/", Sys.Date(), "_nba_predictions.csv"))


#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
library(webshot2)
library(kableExtra)
# Rename columns for better readability
todays_predictions <- todays_predictions %>%
  rename(
    `Game Date & Time` = date,
    `Home Team` = home,
    `Home Score` = home_score,
    `Away Team` = away,
    `Away Score` = away_score,
    `Total Score` = total_score,
    `Spread` = spread,
    `Winner` = winner,
    `Home Win Probability` = home_win_prob,
    `Away Win Probability` = away_win_prob,
    `Home Fair Odds` = home_fair_odds,
    `Away Fair Odds` = away_fair_odds
  )
# Create the static table
static_table <- todays_predictions %>%
  kable(
    format = "html",
    caption = paste0("Today's NBA Game Predictions ", Sys.Date()),
    align = "c",
    escape = TRUE,
    booktabs = TRUE
  ) %>%
  kable_styling(
    bootstrap_options = c("striped"),
    full_width = FALSE,
    position = "center",
    font_size = 12
  ) %>%
  row_spec(0, bold = TRUE) %>%
  column_spec(1, width = "3cm") %>%
  column_spec(2:3, width = "2cm") %>%
  column_spec(4:5, width = "2cm") %>%
  column_spec(6, width = "2cm") %>%
  column_spec(7, width = "2cm") %>%
  column_spec(8, width = "2cm") %>%
  column_spec(9:10, width = "3cm") %>%
  column_spec(11:12, width = "3cm")
# Save as HTML
save_kable(static_table, "assets/todays_nba_predictions.html")
# Convert to PNG
webshot(
  url = "assets/todays_nba_predictions.html",
  file = "assets/todays_nba_predictions.png",
  vwidth = 1200,
  vheight = 800,
  zoom = 2
)
# Optional: Open the image automatically (works on most systems)
browseURL("assets/todays_nba_predictions.png")
#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#