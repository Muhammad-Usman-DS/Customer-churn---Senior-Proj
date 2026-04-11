# Telco Customer Churn Prediction

### Predicting which customers will leave — before they do.

**Live demo:** [ChurnIQ App](https://customer-churn-senior-proj.vercel.app) · [Backend API](https://telco-fastapi.onrender.com) · [API Docs](https://telco-fastapi.onrender.com/docs)

---

## Executive Summary

This project builds a machine learning system that predicts whether a telecom customer is at risk of cancelling their subscription (churning). Using a dataset of 7,043 real customers, we trained an XGBoost model that correctly identifies **82.1% of all customers who will churn**, giving retention teams a prioritised list of at-risk customers to contact proactively. The system is deployed as a live web application with a custom React frontend and a FastAPI backend hosted on Render, accessible to any analyst with a browser — no coding required.

---

## The Business Problem

Customer churn — when a subscriber cancels their service — is one of the most expensive problems a telecom company faces. Acquiring a new customer costs **five to seven times more** than retaining an existing one. In a dataset representative of a mid-size carrier, 26.5% of customers churned within the observation period. On a customer base of hundreds of thousands, even a 1% reduction in churn rate translates to millions of dollars in preserved annual revenue.

The traditional approach is reactive: a customer calls to cancel, and the retention team scrambles to offer a discount. By then it is often too late — the customer has already made up their mind. **The goal of this project is to be proactive**: predict which customers are likely to leave in the near future so that the business can reach out to them first, with the right offer, at the right time.

> **Plain English:** We want to flag at-risk customers *before* they leave, not after.

---

## Our Solution

We built an end-to-end machine learning pipeline that:

1. Ingests raw customer data (demographics, services subscribed, contract type, billing details).
2. Automatically validates, cleans, and transforms that data.
3. Runs each customer through a trained XGBoost classifier to produce a **churn probability score** and a **risk level** (High / Medium / Low).
4. Serves those predictions through a REST API called by a custom React web application in real time.

A business analyst can open the app, type in a customer's details, and get an instant risk assessment — no spreadsheets, no waiting, no data science knowledge required. They can also upload a CSV of hundreds of customers and get bulk predictions in seconds.

---

## Dataset Overview

| Property | Detail |
|---|---|
| **Source** | IBM Telco Customer Churn — [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| **Size** | 7,043 customers, 21 original features |
| **Target variable** | `Churn` — Yes / No (did the customer leave?) |
| **Churn rate** | 26.5% (1,869 churners out of 7,043 customers) |
| **Train / Test split** | 80% training (5,634 customers) / 20% testing (1,409 customers), stratified |

**Key features in the dataset:**

| Category | Features |
|---|---|
| Demographics | Gender, Senior Citizen, Partner, Dependents |
| Phone Services | Phone Service, Multiple Lines |
| Internet Services | Internet Service type, Online Security, Online Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies |
| Account Info | Contract type, Paperless Billing, Payment Method |
| Financials | Tenure (months), Monthly Charges, Total Charges |

**Notable patterns discovered during exploratory analysis:**
- Month-to-month contract customers churn at **42.7%** vs 2.8% for two-year contracts.
- Fiber optic internet customers churn at **41.9%** — likely driven by higher monthly charges.
- Customers without online security or tech support are significantly more likely to churn.
- Electronic check is the most common payment method among churners.
- Long-tenure customers (60+ months) have very low churn rates — loyalty builds retention.

---

## Project Workflow

A step-by-step view of how raw data becomes a live prediction:

```
Step 1: DATA COLLECTION
  └── Raw CSV downloaded from Kaggle
        (7,043 rows, 21 columns, including the Churn label)
          |
          v
Step 2: DATA VALIDATION
  └── Great Expectations checks run automatically:
      - No null values in critical columns
      - TotalCharges is numeric (not a string)
      - Churn column contains only Yes/No
      - All expected column names are present
      - Row count is above minimum threshold
          |
          v
Step 3: PREPROCESSING
  └── src/data/preprocess.py:
      - Strip whitespace from column names
      - Convert TotalCharges from string to float
      - Drop CustomerID (not predictive)
      - Encode binary target: Yes → 1, No → 0
          |
          v
Step 4: FEATURE ENGINEERING
  └── src/features/build_features.py:
      - Binary encoding for 2-category columns (Gender, Partner, etc.)
      - One-hot encoding for multi-category columns (Contract, InternetService, etc.)
      - VIF analysis to detect and remove multicollinear features
      - Result: 30 engineered numeric features ready for training
          |
          v
Step 5: MODEL TRAINING
  └── XGBClassifier with scale_pos_weight to handle 26.5% class imbalance
      - 80/20 stratified train/test split
      - All parameters and metrics logged to MLflow
          |
          v
Step 6: HYPERPARAMETER TUNING
  └── Optuna (20 trials, Recall-optimised):
      - Tunes: n_estimators, learning_rate, max_depth, subsample, colsample_bytree
      - Best hyperparameters re-used for final model training
          |
          v
Step 7: EVALUATION
  └── Predictions at threshold 0.35:
      Recall = 82.1% | Precision = 49.0% | F1 = 61.4% | ROC-AUC = 83.7%
      All results logged as artifacts in MLflow
          |
          v
Step 8: SERVING
  └── FastAPI REST API (src/app/main.py):
      - POST /predict            → single customer prediction
      - POST /predict-batch      → bulk CSV prediction
      - GET  /feature-importance → for Insights page chart
      - GET  /model-stats        → for Insights page metrics
          |
          v
Step 9: FRONTEND
  └── React app (5 pages):
      Predict → BulkPredict → History → Insights → Dataset
      Deployed on Vercel; calls Cloud Run backend via HTTPS
```

---

## Data Preparation & Validation

### Preprocessing (`src/data/preprocess.py`)

Raw data from Kaggle is rarely clean enough to feed directly into a model. Before training, we applied several cleaning steps:

- **Dropped the `CustomerID` column** — a unique identifier has no predictive power and would only confuse the model.
- **Converted `TotalCharges` from string to float** — this column was stored as text in the raw CSV and contained some whitespace characters where charges were missing; those cases were coerced to `0`.
- **Encoded the binary target** — the `Churn` column (Yes/No) was converted to 1/0 so the model can work with numbers.
- **Stripped whitespace** from column names to prevent subtle downstream bugs.

### Data Validation (`src/utils/validate_data.py` — Great Expectations)

**Great Expectations** is a Python library for automated data quality testing. Think of it as a checklist that runs every single time training begins, verifying that the data meets a set of defined rules before any model training happens.

We defined the following expectations for the Telco dataset:

- The dataset must contain exactly the expected column names.
- Critical columns (`tenure`, `MonthlyCharges`) must have no null values.
- The `Churn` column must only contain `0` or `1` after encoding.
- `TotalCharges` must be numeric after conversion.
- Row count must exceed a minimum threshold (to catch accidental empty uploads).

If any expectation fails, the pipeline halts and logs the failures to MLflow as an artifact before raising an error. This prevents a corrupt or incomplete dataset from silently producing a broken model.

> **Business value:** Without validation, a data pipeline bug — such as a blank CSV column — could train a model that gives every customer a 0% churn score, and no one would know until customers were already leaving. Great Expectations is the safety net.

---

## Feature Engineering

Raw customer data does not go into a machine learning model as-is. The model requires all inputs to be **numbers**. Feature engineering is the process of converting raw data into the numerical format the model needs, while preserving as much meaningful information as possible.

### Binary Encoding (2-category columns)

For columns with exactly two possible values, we applied a deterministic 0/1 mapping defined in `BINARY_MAP` inside `src/serving/inference.py`. The exact same mapping is applied at training time (`build_features.py`) and at serving time (`inference.py`), guaranteeing consistency:

| Column | Mapping |
|---|---|
| `gender` | Female → 0, Male → 1 |
| `Partner` | No → 0, Yes → 1 |
| `Dependents` | No → 0, Yes → 1 |
| `PhoneService` | No → 0, Yes → 1 |
| `PaperlessBilling` | No → 0, Yes → 1 |

### One-Hot Encoding (multi-category columns)

For columns with three or more categories (e.g., `Contract`: Month-to-month, One year, Two year), we used **one-hot encoding with `drop_first=True`**. This creates a new binary column for each category, except the first (which is implied when all others are 0), preventing a statistical problem called the **dummy variable trap**.

> **Business analogy:** Instead of storing "Contract = Month-to-month" as the number 1 (which would imply Month-to-month is somehow *greater than* One year), we create two new columns: `Contract_One_year` and `Contract_Two_year`. If a customer has a two-year contract, `Contract_Two_year = 1` and `Contract_One_year = 0`. If they have month-to-month, both are 0. The model reads these as independent signals, not an ordered scale.

### VIF Analysis (Multicollinearity Detection)

**Variance Inflation Factor (VIF)** is a statistical measure that detects when two features are so correlated with each other that they effectively carry the same information. For example, `TotalCharges` is approximately equal to `MonthlyCharges × tenure` — these three columns are highly correlated.

We ran VIF analysis and collapsed or removed redundant features before training. Keeping highly correlated features does not improve a model; it adds noise, inflates apparent importance of duplicated signals, and can slow convergence.

> **Business analogy:** If you hand a hiring committee both a candidate's total years of experience and a year-by-year employment history, they are reading the same information twice. It wastes their attention without adding value. We removed that redundancy from the model's inputs.

The result of feature engineering is **30 numeric features** — expanded from the original 21 through encoding, and refined through VIF filtering.

---

## Handling Class Imbalance

### What is class imbalance?

In our dataset, **73.5% of customers did not churn** and only **26.5% did churn**. This is called a class imbalance. If you train a model on imbalanced data without any adjustment, the model learns that predicting "not churning" for every single customer is a winning strategy — it would be 73.5% accurate while being completely useless for actually finding churners.

### Why it matters for business

A model that never predicts churn gives zero business value. We need the model to be sensitive to the minority class (churners), even if that comes at the cost of some false alarms.

### What we did: `scale_pos_weight`

We used the **`scale_pos_weight`** parameter in XGBoost, which tells the model to penalise missed churners more heavily than missed non-churners during training. The weight is calculated from the class ratio in the training set:

```
scale_pos_weight = count(non-churners) / count(churners)
                 = 4,139 / 1,495  ≈  2.77
```

This means that during training, every time the model misclassifies a churner, that mistake is treated as 2.77 times as costly as misclassifying a non-churner. This pushes the model to look harder for churn signals in the data.

### What we did: Threshold lowering

By default, a classifier only predicts "churn" if the computed probability is above 50%. We **lowered this threshold from 0.50 to 0.35**. At 0.35, the model is more willing to raise a churn flag even when it is only moderately confident. This catches more real churners at the cost of more false alarms. The threshold was selected to balance the business cost of each error type — see the dedicated Recall section below.

---

## Model Selection & Comparison

We trained three candidate models on the same training data and evaluated them on the held-out test set at the 0.35 threshold:

| Model | Recall | Precision | F1 Score | ROC-AUC | Notes |
|---|---|---|---|---|---|
| Logistic Regression | 76.4% | 44.2% | 55.9% | 78.1% | Fast, interpretable, assumes linear relationships between features |
| Random Forest | 79.1% | 46.8% | 58.7% | 80.3% | Strong ensemble, handles non-linearity, but less precise imbalance handling |
| **XGBoost** | **82.1%** | **49.0%** | **61.4%** | **83.7%** | Best on all metrics — selected as the production model |

**Why XGBoost won:**

XGBoost (eXtreme Gradient Boosting) is a tree-based ensemble model that builds hundreds of decision trees **sequentially**, with each new tree learning specifically from the mistakes made by all previous trees. In plain business terms: it iteratively corrects its own errors, making it very good at finding complex, non-linear patterns in tabular data like customer records.

It outperformed Logistic Regression because customer churn is driven by *combinations* of factors (e.g., high charges AND short tenure AND month-to-month contract all together), not simple linear relationships. It outperformed Random Forest because of its sequential boosting approach, built-in regularisation, and superior integration of `scale_pos_weight` for class imbalance handling.

---

## Hyperparameter Tuning

### What is hyperparameter tuning?

A machine learning model has settings that control *how* it learns — these are called **hyperparameters**. Examples include: how many decision trees to build, how deep each tree is allowed to grow, and how aggressively the model corrects its errors with each new tree. These settings cannot be learned automatically from data; they must be chosen before training begins.

**Optuna** is a Python library that automates the search for the best hyperparameter combination. Rather than manually trying combinations, Optuna uses a smart algorithm (TPE — Tree-structured Parzen Estimators) that learns from previous trials to decide which combinations are most promising to explore next.

> **Analogy:** Manually trying hyperparameters is like adjusting a car's engine settings by trying every possible combination. Optuna is like a mechanic who listens to the engine after each test and makes an educated decision about what to adjust next — much faster and smarter.

### Our tuning setup

| Setting | Value |
|---|---|
| Number of trials | 20 |
| Optimisation target | **Recall** (maximise churners caught) |
| Validation method | 3-fold cross-validation per trial |

### Parameters tuned

| Parameter | What it controls | Range searched |
|---|---|---|
| `n_estimators` | Number of trees in the ensemble | 100 – 500 |
| `learning_rate` | How much each tree corrects previous errors | 0.01 – 0.30 |
| `max_depth` | Maximum depth of each decision tree | 3 – 9 |
| `subsample` | Fraction of training rows used per tree | 0.6 – 1.0 |
| `colsample_bytree` | Fraction of features considered per tree | 0.6 – 1.0 |

**Best hyperparameters found by Optuna:**

```
n_estimators     = 301
learning_rate    = 0.034
max_depth        = 7
subsample        = 0.95
colsample_bytree = 0.98
```

### Why we tuned for Recall, not Accuracy

Accuracy is a misleading metric when one type of error costs the business much more than the other. Tuning for Accuracy would reward the model for being right on the easy majority (non-churners) while tolerating a high miss rate on churners. Tuning for Recall directly tells Optuna: "the goal is to catch as many churners as possible." See the next section for the full business reasoning.

---

## Model Performance & Business Interpretation

Final model metrics at threshold 0.35, evaluated on the held-out test set (1,409 customers):

| Metric | Value | What it means in business terms |
|---|---|---|
| **Recall** | **82.1%** | Of every 100 customers who will churn, the model correctly flags 82. Only 18 slip through undetected. |
| **Precision** | **49.0%** | Of every 100 customers the model flags as churners, 49 will actually churn. The other 51 are false alarms. |
| **F1 Score** | **61.4%** | The harmonic mean of Recall and Precision. Useful for comparing models against each other on a single number. |
| **ROC-AUC** | **83.7%** | If you pick a random churner and a random non-churner, the model will rank the churner as higher risk 83.7% of the time. A score of 50% is random guessing; 100% is perfect. |

**Estimated confusion matrix on the test set (1,409 customers):**

| | Predicted: Will Churn | Predicted: Will Stay |
|---|---|---|
| **Actually Churned** | ~308 caught (True Positives) | ~67 missed (False Negatives) |
| **Actually Stayed** | ~321 false alarms (False Positives) | ~713 correctly retained (True Negatives) |

> **Plain English:** The model catches roughly 308 out of 375 actual churners. The retention team receives a list of ~629 names — 308 real churners and 321 customers who would have stayed anyway. That is a ~49% "hit rate" on the contact list. Without the model, the team would need to contact all 1,409 customers to find 375 churners — a 26.6% hit rate. **The model makes retention outreach approximately 1.84x more efficient.**

---

## Why Recall over Accuracy?

This is one of the most important design decisions in the project and deserves a dedicated explanation.

### The core principle: not all errors are equal

Consider the two types of errors this model can make:

**Error Type 1 — Missing a churner (False Negative)**
The model predicts "this customer is fine" but they actually leave. The business loses that customer's entire future lifetime value — potentially hundreds of dollars per year in subscription revenue, compounded over multiple years. This error is **very expensive**.

**Error Type 2 — False alarm (False Positive)**
The model flags a customer as at-risk, but they were going to stay anyway. The retention team contacts them — perhaps offering a courtesy call or a small discount. The business spends the cost of one retention interaction, perhaps a few dollars in agent time. This error is **cheap**.

### The mathematics of the trade-off

At default threshold 0.5, assume the model catches 65% of churners and misses 35%.
At our threshold 0.35, the model catches 82% of churners and misses 18%.

On a real customer base of 100,000 with 26,500 churners at $50/month average value:

| Scenario | Churners missed | Annual revenue lost | Extra false alarm calls | Cost of calls |
|---|---|---|---|---|
| Threshold 0.50 | 9,275 | ~$55.7M | Baseline | Baseline |
| **Threshold 0.35** | **4,770** | **~$28.6M** | +2,000–3,000 | ~$15,000–$30,000 |

The additional retention calls at the lower threshold cost tens of thousands of dollars. The additional churners caught represent tens of millions in preserved revenue. **The math is not close.** A false alarm is always cheaper than a missed churner.

### Why not lower the threshold even further?

At very low thresholds (e.g., 0.20), Recall approaches 100% but Precision collapses — the model would flag nearly everyone as a churner. The retention team becomes overwhelmed, the cost of outreach balloons, and the signal-to-noise ratio deteriorates so badly that analysts stop trusting the model. **0.35 was selected as the point where each additional churner caught still delivers clear ROI**, without flooding the retention team with useless contacts.

---

## The Frontend Application

The React frontend provides five pages, each serving a distinct business function. The app is built with Vite, styled with Tailwind CSS, and uses Axios for API calls to the FastAPI backend.

### 1. Predict Page

The core tool for a retention analyst. The analyst enters a single customer's details across four grouped sections (Demographics, Phone Services, Internet Services, Account) and clicks **Predict Churn**. The result panel shows:

- A **visual risk gauge** — an SVG arc that fills proportionally to the churn probability percentage (e.g., 73.2%)
- A **risk level badge**: High (red), Medium (amber), or Low (green)
- A plain-English **verdict**: "Likely to churn" or "Not likely to churn"

Every prediction is automatically saved to the browser's local storage (up to 200 entries) and appears in the History page.

**Business value:** A retention agent can assess any customer's churn risk in under 30 seconds, from any computer, without needing to query a database or involve a data scientist.

### 2. Bulk Predict Page

Upload a CSV file containing multiple customers (with the same 19 columns as the single predict form). The backend processes every row through the same inference pipeline and returns:

- Summary cards: total customers processed, broken down by High / Medium / Low risk count
- A scrollable results table with contract type, internet service, tenure, monthly charges, probability, and risk badge for each customer
- A **Download CSV** button to export the full scored dataset for CRM import or management reporting

**Business value:** A data analyst can run a weekly batch scoring job — export a segment from the CRM, upload it here, download the risk-scored list, and hand it directly to the retention team. No code required.

### 3. History Page

A log of all past single predictions made in the current browser session, stored in `localStorage` — no server-side database required. Features:

- Filter by risk level (All / High / Medium / Low)
- Full table with timestamp, contract type, internet service, tenure, monthly charges, probability, risk badge, and verdict
- Export the currently filtered view to CSV
- Clear all history with a confirmation prompt

**Business value:** Provides an auditable record of all assessments made during a work session, useful for handoff notes or end-of-day reports, without any backend storage overhead.

### 4. Insights Page

A live model dashboard pulled directly from the API at page load. Displays:

- Four metric cards: Recall, ROC-AUC, F1 Score, Precision — sourced from `model_metrics.json` via the `/model-stats` endpoint
- A model info panel: algorithm name, classification threshold, number of training samples, training set churn rate, and an explanatory note
- An interactive **feature importance bar chart** powered by XGBoost's built-in importance scores, sourced via the `/feature-importance` endpoint, with a dropdown to show the top 10, 15, 20, or all features

**Business value:** Gives managers and stakeholders full transparency into the model's performance and which customer attributes it considers most important — no black box. A stakeholder can immediately see that contract type and tenure are the top predictors of churn, and use that to inform product and pricing strategy.

### 5. Dataset Page

A static overview of the training data, with hardcoded figures sourced from EDA. Contains:

- Six summary statistics: total customers, number of features, churn rate, average tenure, average monthly charge, percentage of senior citizens
- A pie chart showing the churn vs retained split (26.5% vs 73.5%)
- A bar chart showing churn rate by contract type (Month-to-month: 42.7%, One year: 11.3%, Two year: 2.8%)
- A bar chart showing churn rate by internet service type (Fiber optic: 41.9%, DSL: 19.0%, No internet: 7.4%)
- A key findings panel with six actionable business observations from the exploratory analysis

**Business value:** Gives any viewer — executive, analyst, or stakeholder — context for where the model's predictions come from, and surfaces actionable segment insights (e.g., "our fiber optic customers churn at 41.9% — should we revisit fiber pricing?").

---

## What-If Analysis — Deep Dive

### What is it?

The Predict page includes a **"What-If Mode"** toggle that appears in the result panel after the first prediction is made. When switched on, the system automatically re-runs the full prediction pipeline every time any input field changes — no button click required.

### How it works technically

The What-If mechanism is implemented in `frontend/src/pages/Predict.jsx`:

```javascript
function handleChange(key, value) {
  const updated = { ...form, [key]: value };
  setForm(updated);
  // In what-if mode, re-run prediction on every change
  if (whatIfMode && result) {
    runPredict(updated);
  }
}
```

Every input field on the form calls `handleChange` when its value changes. In normal mode, this only updates the React form state. In What-If mode, it additionally calls `runPredict(updated)` immediately — which sends the *updated* customer data to the FastAPI `/predict` endpoint via an HTTP POST request (using Axios from `frontend/src/api.js`).

On the backend, the full inference pipeline runs:
1. `inference.py` receives the raw 19-field customer dictionary.
2. `_serve_transform()` applies binary encoding, one-hot encoding, and feature alignment to produce a 30-feature array in the exact order the model expects.
3. `sklearn_model.predict_proba()` scores the input and returns a churn probability.
4. The probability is compared against the 0.35 threshold to produce a binary prediction.
5. The risk level (High ≥ 65%, Medium ≥ 35%, Low < 35%) is assigned.
6. The result JSON is returned to the frontend.

The result panel updates with the new probability and risk gauge in real time. The entire round trip — from changing a dropdown to seeing the updated probability on screen — takes under one second on a production connection.

### Why it is valuable for business analysts

Traditional model deployments require a "submit" action for each scenario. What-If mode turns the prediction tool into an interactive **retention strategy simulator**. Instead of asking "will this customer churn?", an analyst can ask "what specific intervention would most effectively reduce this customer's risk?" — and get an answer instantly.

**Example scenarios an analyst can explore:**

| Business Question | How to explore it in What-If mode |
|---|---|
| "If this customer upgrades from Month-to-month to a Two-year contract, does their risk drop below High?" | Change the Contract dropdown — result updates instantly |
| "At what monthly charge level does this customer tip into High risk?" | Increment the Monthly Charges field and watch the probability gauge |
| "How much does adding Online Security change the churn probability?" | Toggle Online Security from No to Yes |
| "Is offering a longer tenure discount more effective than a contract change for this customer?" | Alternate between changing tenure and contract — compare the probability delta |
| "Would this senior citizen on electronic check be lower risk if they switched to automatic payment?" | Change Payment Method and observe the shift |

This feature elevates the tool from a simple prediction endpoint to a genuine **business decision-support system** — enabling evidence-based retention strategy at the individual customer level.

---

## Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Data processing** | Pandas, NumPy | Data loading, cleaning, transformation |
| **Data validation** | Great Expectations | Automated data quality checks before training |
| **Feature engineering** | Pandas, scikit-learn | Binary encoding, one-hot encoding, VIF analysis |
| **ML model** | XGBoost | Gradient boosted tree ensemble classifier |
| **Hyperparameter tuning** | Optuna | Automated search over hyperparameter space (20 trials) |
| **Experiment tracking** | MLflow | Logs parameters, metrics, model artifacts, and reports |
| **API framework** | FastAPI | REST API serving predictions and model metadata |
| **API server** | Uvicorn | ASGI server running the FastAPI application |
| **Frontend framework** | React (Vite) | Five-page single-page web application |
| **Frontend styling** | Tailwind CSS | Utility-first CSS framework |
| **Frontend charts** | Recharts | Interactive bar and pie charts on Insights and Dataset pages |
| **HTTP client** | Axios | Frontend-to-backend API communication |
| **Containerisation** | Docker | Reproducible, portable deployment environment |
| **Container registry** | Docker Hub | Stores and serves built Docker images |
| **Backend hosting** | Render | Container hosting for the FastAPI backend |
| **Frontend hosting** | Vercel | Global CDN hosting for the React app |
| **CI/CD** | GitHub Actions | Automated Docker build and push on every push to `main` |

---

## Experiment Tracking with MLflow

**MLflow** is an open-source platform for managing the machine learning lifecycle. Think of it as a **lab notebook that writes itself**: every time an experiment runs, MLflow automatically records what parameters were used, what metrics resulted, and which model artifact was produced.

Every run of `scripts/run_pipeline.py` logs to MLflow:

- **Parameters:** model type, classification threshold (0.35), test split ratio (0.20)
- **Metrics:** Precision, Recall, F1, ROC-AUC, training time (seconds), inference time (seconds), data quality pass/fail flag
- **Artifacts:** trained model in MLflow's standard sklearn format, feature column list (`feature_columns.txt`), preprocessing metadata (`preprocessing.pkl`), failed validation report (if applicable)

The MLflow UI (`mlflow ui`) presents all historical runs in a comparison table, enabling side-by-side analysis of how different hyperparameter combinations or preprocessing choices affected model performance.

The `mlruns/` directory is **gitignored** — it is large and machine-specific. For production, the final trained model is saved separately in the `model/` folder, which is included in the Docker container image at build time. This makes the model available in production without requiring a running MLflow tracking server.

---

## CI/CD & Deployment

### GitHub Actions (`/.github/workflows/ci.yml`)

Every push to the `main` branch automatically triggers the following pipeline:

1. **Checkout** — GitHub Actions pulls the latest code from the repository.
2. **Docker Buildx setup** — configures the modern Docker builder, which supports multi-platform image builds.
3. **Docker Hub login** — authenticates using encrypted repository secrets (`DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`).
4. **Build and push** — builds the Docker image from `dockerfile` and pushes it to `usmangoat/telco-fastapi:latest` on Docker Hub.

Every merged change is automatically packaged and published — no manual build steps, no "it works on my machine" problems.

### Render (Backend)

Render is a cloud platform that runs the Docker container from Docker Hub. Key properties:

- Pulls `usmangoat/telco-fastapi:latest` from Docker Hub and runs it as a web service
- Exposes the FastAPI application on a public HTTPS URL: `https://telco-fastapi.onrender.com`
- Free tier sleeps after 15 minutes of inactivity — first request after sleep takes ~30 seconds to wake up
- The frontend's environment variable `VITE_API_URL` points to this URL

### Vercel (Frontend)

The React app (built with Vite) is deployed on Vercel. Vercel detects the `frontend/` directory, runs `npm run build` to produce an optimised static bundle in `dist/`, and serves it instantly from a global CDN. Every push to `main` also triggers an automatic Vercel redeploy.

---

## How to Run Locally

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker (optional, for containerised backend run)

### Backend

```bash
# 1. Clone the repository
git clone https://github.com/Muhammad-Usman-DS/telco-customer-churn.git
cd "Customer churn - Senior Proj"

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. (Optional) Retrain the model
python scripts/run_pipeline.py \
    --input data/WA_Fn-UseC_-Telco-Customer-Churn.csv \
    --target Churn

# 4. Start the FastAPI server
uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000

# API available at:    http://localhost:8000
# Interactive docs at: http://localhost:8000/docs
```

### Frontend

```bash
# In a new terminal window
cd frontend

# Install dependencies
npm install

# (Optional) Set backend URL — defaults to http://localhost:8000 if unset
echo "VITE_API_URL=http://localhost:8000" > .env

# Start the development server
npm run dev

# Frontend available at: http://localhost:5173
```

### Docker (containerised backend)

```bash
# Build the Docker image
docker build -t telco-churn-api .

# Run the container
docker run -p 8000:8000 telco-churn-api

# API available at: http://localhost:8000
```

---

*Built by Syed Muhammad Usman — Senior Project, 2025.*
