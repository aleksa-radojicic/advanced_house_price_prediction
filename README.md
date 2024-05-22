# Advanced House Price Prediction using Machine Learning Algorithms ##
This is my first modular Data Science project, structured similarly to the CookieCutter Data Science template. The goal was to develop various ML models with default hyperparameters using pipelines to predict house prices in Ames, Iowa, USA. The dataset is a part of the Kaggle competition [*House Prices - Advanced Regression Techniques*](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/).

## File Descriptions
* **train.csv** - the training set (1460 rows);
* **test.csv** - the test set (1459 rows);
* **data_description.txt** - full description of each column, originally prepared by Dean De Cock but lightly edited to match the column names used here (79 explanatory variables);
* **sample_submission.csv** - a benchmark submission from a linear regression on year and month of sale, lot square footage, and number of bedrooms.

## Preprocessing
The key steps performed in preprocessing are as follows:
* Transformations from Univariate Analysis;
* Transformations from Multivariate Analysis;
* Feature Engineering;
* Post Feature Engineering;
* Dropping columns scheduled for deletion;
* Columns type prefixing;
* Columns transformations.

## Modelling
For model evaluation, 20% was used as the evaluation set [eval set], while the remaining 80% was used for training ML models with default hyperparameters. The target variable was log-transformed during modelling to further improve model performance. A total of 256 features (including derived columns) were used for predicting the output. The evaluation metric used was coefficient of determination (R2).

Column transformations performed for tree-based ML models:
* numerical: "passthrough";
* nominal: OrdinalEncoder.

Column transformations performed for non-tree-based ML models:
* numerical: RobustScaler;
* nominal: OneHotEncoder.

## Results on Evaluation Set
| Model | R2      |
|-------|---------|
| Ridge | **0.14028** |
| KNN   | 0.21499 |
| SVR   | 0.33440 |
| RF    | 0.15327 |
| XGB   | 0.15700 |
| Ada   | 0.16507 |
| DT    | 0.20878 |

Ridge regression performed the best on the eval set, suggesting that a more complex ML model will likely overfit. On the train set, the R2 was 0.09952, indicating overfitting.

The entire dataset ('train.csv') was afterwards used to train the Ridge regression. A 'submission.csv' file was created with predictions for the test set ('test.csv') and submitted to the Kaggle competition. 

## Submission Score
* Ridge : **0.13680**;
* Leaderboard Position: **1508**.