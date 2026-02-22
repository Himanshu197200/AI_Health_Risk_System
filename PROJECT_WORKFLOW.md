# 📘 Project Workflow — How Everything Works

This document explains the complete working of the **Intelligent Patient Risk Assessment System**, file by file, and how data flows through the pipeline.

---

## 🔁 High-Level Workflow

```
raw_data.csv
     │
     ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ preprocess.py │ ──▶ │ features.py  │ ──▶ │   train.py   │
│              │     │              │     │              │
│ • Load CSV   │     │ • Fit scaler │     │ • Linear Reg │
│ • Select 10  │     │ • Scale data │     │ • Dec Tree   │
│   features   │     │ • Save scaler│     │ • Save models│
│ • Encode     │     └──────────────┘     └──────────────┘
│ • Split data │                                │
└──────────────┘                                ▼
                                        ┌──────────────┐
                                        │ evaluate.py  │
                                        │              │
                                        │ • MAE, MSE   │
                                        │ • RMSE, R²   │
                                        │ • Pick best  │
                                        └──────────────┘
                                                │
                                                ▼
                                        ┌──────────────┐
                                        │  models/     │
                                        │              │
                                        │ • best_heart │
                                        │ • best_diab  │
                                        │ • scaler.pkl │
                                        │ • metrics    │
                                        └──────────────┘
                                                │
                                                ▼
                                        ┌──────────────┐     ┌──────────────┐
                                        │  predict.py  │ ◀── │ streamlit_app│
                                        │              │     │              │
                                        │ • Load model │     │ • User input │
                                        │ • Scale input│     │ • Show score │
                                        │ • Predict    │     │ • Show level │
                                        └──────────────┘     └──────────────┘
```

---

## 📂 File-by-File Explanation

### 1. `data/raw_data.csv`

**What it is:** The original patient dataset with ~97,000 rows.

**Contains:**
- 28 input features (demographics, lifestyle, clinical measurements)
- 5 target risk scores (heart, diabetes, hypertension, obesity, cholesterol)

We only use **heart_disease_risk_score** and **diabetes_risk_score** as targets.

---

### 2. `src/preprocess.py` — Data Preparation

**Purpose:** First step — loads raw data and prepares it for training.

**What it does (step by step):**

1. **`load_dataset(filepath)`** — Reads the CSV file into a Pandas DataFrame
2. **`select_features(df)`** — Drops unnecessary columns, keeps only 10 important features + 2 targets. This was decided after running a RandomForest feature importance analysis that found 18+ features had <0.5% importance
3. **`encode_categoricals(df)`** — Converts `smoking_status` (text: Never/Former/Current) into numeric columns using one-hot encoding (e.g. `smoking_status_Former`, `smoking_status_Never`)
4. **`separate_features_targets(df)`** — Splits the DataFrame into:
   - `X` — feature matrix (what the model learns from)
   - `y_heart` — heart disease risk scores (what we predict)
   - `y_diabetes` — diabetes risk scores (what we predict)
5. **`split_data(X, y)`** — Splits data into 80% training and 20% testing

**Selected 10 Features:**
| Feature | Why It Matters |
|---|---|
| Age | Strongest predictor for heart risk |
| bmi | Strongest predictor for diabetes risk |
| glucose_fasting | Key diabetes indicator |
| hba1c | Long-term blood sugar control |
| systolic_bp | Blood pressure — heart risk |
| ldl_cholesterol | "Bad" cholesterol — heart risk |
| hdl_cholesterol | "Good" cholesterol — protective |
| smoking_status | Major risk factor for heart disease |
| physical_activity_minutes_per_week | Protective lifestyle factor |
| family_history_diabetes | Genetic predisposition |

---

### 3. `src/features.py` — Feature Scaling

**Purpose:** Normalizes the data so all features are on the same scale.

**Why this matters:** Without scaling, features with large values (like `glucose_fasting ≈ 60–300`) would dominate features with small values (like `family_history_diabetes = 0 or 1`).

**What it does:**

1. **`fit_scaler(X_train)`** — Creates a `StandardScaler` and learns the mean and standard deviation from training data **only** (to prevent data leakage)
2. **`transform_features(scaler, X)`** — Transforms any data using the learned mean/std: `(value - mean) / std`
3. **`save_scaler(scaler)`** — Saves the fitted scaler to `models/scaler.pkl` so it can be reused during prediction
4. **`save_feature_columns(columns)`** — Saves the column names/order to `models/feature_columns.pkl` so inference inputs can be aligned to the same schema

---

### 4. `src/train.py` — Model Training

**Purpose:** Trains two different regression models.

**Models used:**

| Model | How It Works |
|---|---|
| **LinearRegression** | Finds the best straight-line relationship between features and target. Simple, fast, interpretable. |
| **DecisionTreeRegressor** | Builds a tree of if-else rules to split data. Can capture non-linear patterns but may overfit. |

**What it does:**

1. **`train_linear_regression(X_train, y_train)`** — Fits a LinearRegression model and returns it
2. **`train_decision_tree(X_train, y_train)`** — Fits a DecisionTreeRegressor model and returns it
3. **`save_model(model, filename)`** — Saves a trained model to `models/` using joblib

Each model is trained **separately** for each disease (heart and diabetes), so we train 4 models total:
- LinearRegression for heart
- DecisionTree for heart
- LinearRegression for diabetes
- DecisionTree for diabetes

---

### 5. `src/evaluate.py` — Model Evaluation

**Purpose:** Measures how good each model is and picks the winner.

**Metrics used:**

| Metric | What It Measures | Good Value |
|---|---|---|
| **MAE** (Mean Absolute Error) | Average error in predictions | Lower is better |
| **MSE** (Mean Squared Error) | Average squared error (penalizes big errors) | Lower is better |
| **RMSE** (Root MSE) | Same as MSE but in original units | Lower is better |
| **R² Score** | How much variance the model explains (0–1) | Higher is better (1 = perfect) |

**What it does:**

1. **`calculate_metrics(y_true, y_pred)`** — Computes all 4 metrics and returns them as a dictionary
2. **`compare_models(metrics_a, metrics_b)`** — Compares two models: picks the one with lower RMSE (ties broken by higher R²)
3. **`print_metrics(metrics, model_name)`** — Pretty-prints metrics to the terminal

---

### 6. `main.py` — The Pipeline Orchestrator

**Purpose:** The single entry point that runs the entire training workflow. You run `python main.py` and everything happens automatically.

**Execution flow:**

```
Step 1: Load raw_data.csv (97,297 rows)
    │
Step 2: Select 10 important features → 12 columns (10 features + 2 targets)
    │
Step 3: One-hot encode smoking_status → 13 columns
    │
Step 4: Separate into X (features) and y (targets)
    │
Step 5: Split into 80% train / 20% test
    │
Step 6: Fit scaler on training data → save scaler.pkl
    │
Step 7: Scale both train and test data
    │
Step 8: Train 2 models for HEART risk
    │   └─ Evaluate both → pick best → save best_heart_model.pkl
    │
Step 9: Train 2 models for DIABETES risk
    │   └─ Evaluate both → pick best → save best_diabetes_model.pkl
    │
Step 10: Save all metrics to metrics.json
    │
Done! All artifacts in models/
```

---

### 7. `src/predict.py` — Inference (Prediction Only)

**Purpose:** Used at prediction time (NOT during training). Loads saved models and makes predictions on new patient data.

**What it does:**

1. **`load_model(path)`** — Loads a saved `.pkl` model from disk
2. **`load_scaler(path)`** — Loads the saved scaler
3. **`predict_risk(input_df, model_path)`** — The core function:
   - Takes a single patient's data as a DataFrame
   - One-hot encodes the `smoking_status` column
   - Aligns columns to match what the model was trained on
   - Scales the input using the saved scaler
   - Runs the model's `.predict()` method
   - Clips the score to 0–100
   - Converts score to risk level (0–33 = Low, 34–66 = Moderate, 67–100 = High)
4. **`predict_heart_risk(input_df)`** — Shortcut that calls `predict_risk` with the heart model
5. **`predict_diabetes_risk(input_df)`** — Shortcut that calls `predict_risk` with the diabetes model

---

### 8. `app/streamlit_app.py` — Web Interface

**Purpose:** Interactive UI that lets users enter patient data and see predictions without touching any code.

**What it does:**

1. Displays 10 input fields matching the selected features
2. On button click, builds a DataFrame from the inputs
3. Calls `predict_heart_risk()` and `predict_diabetes_risk()` from `predict.py`
4. Displays the risk scores and colored risk level badges
5. Shows a **sidebar** with model performance comparison (loaded from `metrics.json`)

**Important:** The app **never retrains** the model. It only loads pre-trained models from `models/`.

---

### 9. `models/` — Saved Artifacts

| File | What It Contains |
|---|---|
| `best_heart_model.pkl` | The best-performing model for heart disease prediction |
| `best_diabetes_model.pkl` | The best-performing model for diabetes prediction |
| `scaler.pkl` | The StandardScaler fitted on training data |
| `feature_columns.pkl` | The ordered list of feature column names after encoding |
| `metrics.json` | All model evaluation metrics (for the UI sidebar) |

---

### 10. `requirements.txt` — Dependencies

Lists all Python packages needed: pandas, numpy, scikit-learn, streamlit, joblib, matplotlib, seaborn.

---

## 🔑 Key Design Principles

1. **Training and inference are completely separated** — `main.py` trains; `predict.py` predicts. They never mix.
2. **Scaler is fit on training data only** — Prevents data leakage (the test set never influences scaling).
3. **Feature columns are saved** — So prediction-time inputs are aligned to the exact same column order used during training.
4. **Best model is auto-selected** — Based on lowest RMSE and highest R², only the winning model is saved.
5. **No global variables** — Everything is passed through function arguments.
6. **joblib for serialization** — Efficient binary format for scikit-learn models.
