# Import libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    TimeSeriesSplit,
    RandomizedSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from flask import Flask, request, jsonify
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# Load dataset
data = pd.read_csv("creditcard.csv")  # Update path

# -------------------------------------------------------------------------------------
# 1. Define Dependent (Target) and Independent Variables (Features)
# -------------------------------------------------------------------------------------
X = data.drop("Class", axis=1)  # Independent variables (all features except 'Class')
y = data["Class"]  # Dependent variable (fraud label)

print("Independent Variables (Features):\n", X.columns.tolist())
print("\nDependent Variable (Target):\n", y.name)

# -------------------------------------------------------------------------------------
# 2. Handle Class Imbalance with SMOTE (Synthetic Minority Oversampling Technique)
# -------------------------------------------------------------------------------------
# Split data first to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Apply SMOTE only to the training data
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Scale features
scaler = StandardScaler()
X_train_bal = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)

# -------------------------------------------------------------------------------------
# 3. Cross-Validation (Stratified K-Fold)
# -------------------------------------------------------------------------------------
# Define models with class weights where applicable
models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=42),
    "XGBoost": XGBClassifier(scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1])),
    "LightGBM": LGBMClassifier(class_weight="balanced", random_state=42),
}

# Evaluate models using cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    pipeline = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("model", model),
    ])
    
    # Cross-validated metrics
    scores = {
        "precision": [],
        "recall": [],
        "f1": [],
        "roc_auc": [],
    }
    
    for train_idx, val_idx in cv.split(X_train_bal, y_train_bal):
        X_fold_train, X_fold_val = X_train_bal[train_idx], X_train_bal[val_idx]
        y_fold_train, y_fold_val = y_train_bal.iloc[train_idx], y_train_bal.iloc[val_idx]
        
        model.fit(X_fold_train, y_fold_train)
        y_pred = model.predict(X_fold_val)
        y_prob = model.predict_proba(X_fold_val)[:, 1]
        
        scores["precision"].append(precision_score(y_fold_val, y_pred))
        scores["recall"].append(recall_score(y_fold_val, y_pred))
        scores["f1"].append(f1_score(y_fold_val, y_pred))
        scores["roc_auc"].append(roc_auc_score(y_fold_val, y_prob))
    
    # Average scores
    results[name] = {
        "Precision (CV)": np.mean(scores["precision"]),
        "Recall (CV)": np.mean(scores["recall"]),
        "F1 (CV)": np.mean(scores["f1"]),
        "ROC-AUC (CV)": np.mean(scores["roc_auc"]),
    }

# Display cross-validation results
results_df = pd.DataFrame(results).T
print("\nCross-Validation Results:\n", results_df)

# -------------------------------------------------------------------------------------
# 4. Hyperparameter Tuning (Example: Random Forest)
# -------------------------------------------------------------------------------------
# Define parameter grid for RandomizedSearchCV
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}

# Initialize RandomizedSearchCV
rf = RandomForestClassifier(class_weight="balanced", random_state=42)
search = RandomizedSearchCV(
    rf,
    param_grid,
    n_iter=10,
    cv=3,
    scoring="roc_auc",
    random_state=42,
    n_jobs=-1,
)
search.fit(X_train_bal, y_train_bal)

# Best model
best_rf = search.best_estimator_
print("\nBest Parameters (Random Forest):\n", search.best_params_)

# -------------------------------------------------------------------------------------
# 5. Backtesting (Time-Based Split)
# -------------------------------------------------------------------------------------
# Sort data by time (assuming 'Time' column exists)
data_sorted = data.sort_values("Time")
X_time = data_sorted.drop("Class", axis=1)
y_time = data_sorted["Class"]

# TimeSeriesSplit (simulate real-world deployment)
tscv = TimeSeriesSplit(n_splits=3)
backtest_scores = []

for train_idx, test_idx in tscv.split(X_time):
    X_train_time, X_test_time = X_time.iloc[train_idx], X_time.iloc[test_idx]
    y_train_time, y_test_time = y_time.iloc[train_idx], y_time.iloc[test_idx]
    
    # Preprocess
    X_train_time_bal, y_train_time_bal = smote.fit_resample(X_train_time, y_train_time)
    scaler_time = StandardScaler()
    X_train_time_bal = scaler_time.fit_transform(X_train_time_bal)
    X_test_time_scaled = scaler_time.transform(X_test_time)
    
    # Train and evaluate
    best_rf.fit(X_train_time_bal, y_train_time_bal)
    y_pred_time = best_rf.predict(X_test_time_scaled)
    roc_auc = roc_auc_score(y_test_time, y_pred_time)
    backtest_scores.append(roc_auc)

print("\nBacktesting ROC-AUC Scores:", backtest_scores)
print("Average Backtesting ROC-AUC:", np.mean(backtest_scores))

# -------------------------------------------------------------------------------------
# 6. Save the Best Model
# -------------------------------------------------------------------------------------
with open("best_fraud_model.pkl", "wb") as f:
    pickle.dump(best_rf, f)

# -------------------------------------------------------------------------------------
# 7. Deploy the Model with Flask (Example)
# -------------------------------------------------------------------------------------
app = Flask(__name__)

# Load the saved model
with open("best_fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    features_scaled = scaler.transform(features)  # Use the original scaler
    prediction = model.predict(features_scaled)
    return jsonify({"fraud_prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
