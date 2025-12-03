from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import metrics
import pandas as pd

"""
Yardage prediction model with stronger feature selection, a clearer pipeline,
and cross‑validated evaluation.

This version does **not** try to construct columns that don't exist in the
CSV. Instead, it uses all meaningful numeric columns and compares a
GradientBoosting model against a simple baseline that always predicts the
median yards gained.
"""

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("Loading data...")
condensed_train = pd.read_csv("condensed_train_clean.csv", low_memory=False)
print(f"Data shape: {condensed_train.shape}")

# ---------------------------------------------------------------------------
# 2. Feature / target definition
# ---------------------------------------------------------------------------
# Columns that are identifiers and should not be used as predictors
id_cols = ["GameId", "PlayId", "NflIdRusher"]

target_col = "Yards"

# Use **all other numeric columns** as features
feature_cols = [
    c for c in condensed_train.columns
    if c not in id_cols + [target_col]
]

print("Using the following feature columns:")
for c in feature_cols:
    print("  -", c)

X = condensed_train[feature_cols].values.astype(float)
y = condensed_train[target_col].values.astype(float)

# ---------------------------------------------------------------------------
# 3. Train / test split
# ---------------------------------------------------------------------------
print("\nSplitting data (80% train / 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# ---------------------------------------------------------------------------
# 4. Build pipelines (baseline + Gradient Boosting)
# ---------------------------------------------------------------------------
# Trees do not need scaling, so the preprocessing is just median imputation.
preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

# Baseline: always predict the median yards
baseline_model = Pipeline([
    ("prep", preprocessor),
    ("model", DummyRegressor(strategy="median")),
])

# Main model: Gradient Boosting with more regularization (to help generalize)
# These hyperparameters are a good starting point; you can tune them further.
boosting_model = Pipeline([
    ("prep", preprocessor),
    ("model", HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.06,
        max_iter=300,
        max_depth=6,
        min_samples_leaf=40,
        l2_regularization=0.1,
        random_state=42,
    )),
])

# ---------------------------------------------------------------------------
# 5. Cross‑validated performance (on the whole dataset)
# ---------------------------------------------------------------------------
print("\nRunning 5‑fold cross‑validation...")
cv = KFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_cv(model, name: str):
    r2_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
    mae_scores = -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
    print(f"\n{name} CV results (5‑fold):")
    print(f"  R2:  mean={r2_scores.mean():.3f},  std={r2_scores.std():.3f}")
    print(f"  MAE: mean={mae_scores.mean():.3f}, std={mae_scores.std():.3f}")

# Baseline vs model
evaluate_cv(baseline_model, "Baseline (median yards)")
evaluate_cv(boosting_model, "Gradient Boosting")

# ---------------------------------------------------------------------------
# 6. Fit final model on the train set and evaluate on the held‑out test set
# ---------------------------------------------------------------------------
print("\nTraining Gradient Boosting model on training set...")
boosting_model.fit(X_train, y_train)

y_pred = boosting_model.predict(X_test)

print("\nModel Performance Metrics on held‑out test set:")
print("Mean Absolute Error:", round(metrics.mean_absolute_error(y_test, y_pred), 2))
print("Mean Squared Error:", round(metrics.mean_squared_error(y_test, y_pred), 2))
print("Root Mean Squared Error:", round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 2))
print("R2 Score:", round(metrics.r2_score(y_test, y_pred), 3))

# ---------------------------------------------------------------------------
# 7. Diagnostic plots
# ---------------------------------------------------------------------------
# Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Actual Yards")
plt.ylabel("Predicted Yards")
plt.title("Actual vs Predicted Yards Gained")
plt.tight_layout()
plt.savefig("yards_prediction_results.png")
plt.close()

# Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color="r", linestyle="--")
plt.xlabel("Predicted Yards")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig("residuals.png")
plt.close()

# Error distribution
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=50, edgecolor="black")
plt.xlabel("Prediction Error (yards)")
plt.ylabel("Frequency")
plt.title("Distribution of Prediction Errors")
plt.tight_layout()
plt.savefig("error_distribution.png")
plt.close()

print("\nError Distribution Statistics:")
print("Mean Error:", round(np.mean(residuals), 2))
print("Median Error:", round(np.median(residuals), 2))
print("Standard Deviation of Error:", round(np.std(residuals), 2))
print("25th Percentile Error:", round(np.percentile(residuals, 25), 2))
print("75th Percentile Error:", round(np.percentile(residuals, 75), 2))

# ---------------------------------------------------------------------------
# 8. Feature importance (using the trained Gradient Boosting model)
# ---------------------------------------------------------------------------
# Compute permutation importance so we always have importances, even if the
# underlying estimator doesn't expose feature_importances_.
print("\nComputing permutation feature importance on test set...")
perm_result = permutation_importance(
    boosting_model,
    X_test,
    y_test,
    n_repeats=5,
    random_state=42,
    n_jobs=-1,
)

fi = perm_result.importances_mean
fi_std = perm_result.importances_std

feature_importance = (
    pd.DataFrame({
        "Feature": feature_cols,
        "Importance": fi,
        "Std": fi_std,
    })
    .sort_values("Importance", ascending=False)
)

print("\nTop 15 Most Important Features (permutation importance):")
print(feature_importance.head(15)[["Feature", "Importance"]].to_string(index=False))

plt.figure(figsize=(12, 6))
sns.barplot(data=feature_importance.head(15), x="Importance", y="Feature")
plt.title("Top 15 Most Important Features (permutation importance)")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()