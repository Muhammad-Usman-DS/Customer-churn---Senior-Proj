# Description and Report of the project.

The final report would be made into a file format and this section would be replaced with a readme file with the description of the project, dataset, and techniques e.t.c

For now, the report and changes made are committed in this readme file for ease of readibility.

## Data Source:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data

### Update as of 24 Februaury:

## Multicollinearity Check

I ran a VIF analysis to check for multicollinearity before moving into modeling. Several features showed strong correlation, especially:

- Internet service related dummy variables

- TotalCharges, MonthlyCharges, and tenure

- Some phone service indicators

To reduce redundancy, I collapsed structurally similar dummy variables and removed dependent columns. This helped clean up the feature space before training.

### Class Imbalance

The churn distribution is:

5174 non-churn

1869 churn

~27% churn rate

Since the imbalance is moderate, I decided not to use oversampling. Instead, handled it using:

Class weighting

Threshold tuning

I lowered the prediction threshold from 0.5 to 0.3 to improve recall, since missing churners is more costly than false positives.

### Train-Test Split

Then performed an 80/20 split with stratification to maintain the churn ratio in both sets.

## Model Comparison

I trained and compared:

1. Logistic Regression

2. Random Forest

3. XGBoost

Evaluation focused on recall and overall classification performance rather than accuracy.

XGBoost performed the best overall:

- Faster training time

- Stronger recall

- Good balance across metrics

At this stage, XGBoost is the leading model moving forward for further tuning.
