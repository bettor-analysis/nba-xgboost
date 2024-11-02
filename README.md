# NBA XGBoost Model for Predicting Team Scores

This project is a predictive model for NBA team scores using historical and current season data. The model is built with the `xgboost` algorithm and uses various team metrics to train and evaluate its performance.

## Overview

This model predicts NBA team scores for scheduled games, calculates win probabilities, and generates fair odds in American format. The predictions are saved as a daily CSV file for reference.

## Setup

1. Clone this repository.
2. Ensure the necessary libraries are installed (see Dependencies below).
3. Create the following folders for data storage:
   - `data/` - for storing historical and current season data.
   - `daily_predictions/` - for saving daily prediction outputs.
   - `assets/` - for saving generated static table images.

## Data Preparation

1. **Load Historical Data**: Load historical NBA data from `data/historical.csv`.
2. **Current Season Data**: Load current season data using the `hoopR` package.
3. **Data Cleaning**: Remove non-relevant teams (e.g., All-Star teams) and bind historical and current data together.
4. **Feature Engineering**:
   - Calculate team metrics such as Effective Field Goal Percentage, True Shooting Percentage, Possessions, and Offensive/Defensive Ratings.
   - Final data is structured to include key features for model training.

## Model Training

1. **Split Data**: Split into training+validation and final test sets.
2. **Define Features and Response**: Select metrics (e.g., `effective_field_goal_percentage`, `true_shooting_percentage`, `OR`, `DR`) as predictors, with `team_score` as the response variable.
3. **Cross-Validation and Hyperparameter Tuning**: Use 5-fold cross-validation with a grid of hyperparameters (learning rate, depth, etc.).
4. **Train the XGBoost Model**: Train with `trainControl` from `caret` package and optimize for minimum RMSE.

## Making Predictions

1. **Generate Predictions**: Use the trained model to predict scores for 2025 teams.
2. **Simulate Final Scores**: Run simulations to get the median score for each team.
3. **Generate Win Probabilities and Fair Odds**: Simulate win probabilities based on score distributions for each team, then calculate American-style fair odds.
4. **Save Predictions**: Write the predictions to a dated CSV file in the `daily_predictions/` folder.

## Daily Predictions Table

The model generates a static HTML and PNG table of daily predictions for easy reference. 
- **HTML File**: `assets/todays_nba_predictions.html`
- **PNG Image**: `assets/todays_nba_predictions.png`

The table includes:
- Home and Away teams with predicted scores
- Total score, spread, winner, win probabilities, and fair odds

## Dependencies

This project relies on the following R packages:
- **tidyverse**: Data manipulation and visualization
- **hoopR**: NBA data access
- **xgboost**: Model training
- **caret**: Model training and evaluation
- **doParallel**: Parallel processing
- **lubridate**: Date handling
- **kableExtra**: HTML table generation
- **webshot2**: HTML to PNG conversion

Ensure all dependencies are installed by running:
```R
install.packages(c("tidyverse", "hoopR", "xgboost", "caret", "doParallel", "lubridate", "kableExtra", "webshot2"))
