from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import metrics
import pandas as pd
import re
import os

# Load the condensed training data with low_memory=False to handle mixed types
print("Loading data...")
condensed_train = pd.read_csv('condensed_train.csv', low_memory=False)

# Use the provided `condensed_train.csv` directly; compute closest defenders on the fly
computed_closest = False

# Select relevant features
# Basic play context features
play_features = ['Quarter', 'Down', 'Distance', 'YardLine', 'DefendersInTheBox']

# Get position-related columns for rusher and closest defenders
pos_cols = ['_X', '_Y', '_S', '_A', '_Dis', '_Dir']
player_cols = []

# Add rusher's metrics (always Rush1_ prefix)
for col in pos_cols:
    player_cols.append('Rush1' + col)

# Add metrics for first 5 defenders (increased from 3)
for i in range(5):
    def_prefix = f'Def{i+1}'
    for col in pos_cols:
        player_cols.append(def_prefix + col)

# Combine all features
feature_cols = play_features + player_cols

# Prepare features and target
print("Preparing features...")

# Precompute which defender indices exist in the DataFrame (e.g. Def1_X, Def2_X, ...)
def_indices = []
for c in condensed_train.columns:
    m = re.match(r"Def(\d+)_X$", c)
    if m:
        def_indices.append(int(m.group(1)))
def_indices = sorted(def_indices)

# Build feature name list (uses "ClosestDefN" slots based on proximity)
feature_names = []
for col in play_features:
    feature_names.append(col)
for p in pos_cols:
    feature_names.append('Rush1' + p)
for slot in range(1, 6):
    for p in pos_cols:
        feature_names.append(f'ClosestDef{slot}_{p}')
    for rel in ['X_Dist', 'Y_Dist', 'S_Diff', 'A_Diff']:
        feature_names.append(f'Rush_ClosestDef{slot}_{rel}')

# We'll compute closest defenders on the fly when constructing features below.
# This script will use the already-created `condensed_train.csv` and will not
# create or write a separate `condensed_train_with_closest.csv` file.
feature_data = []
for _, row in condensed_train.iterrows():
    features = []
    # Add play features
    for col in play_features:
        features.append(row.get(col, np.nan))

    # Rusher features - for X/Y use the canonical 'X'/'Y' columns as base, otherwise prefer Rush1_* then fallback
    for p in pos_cols:
        if p == '_X':
            features.append(row.get('X', row.get('Rush1_X', np.nan)))
        elif p == '_Y':
            features.append(row.get('Y', row.get('Rush1_Y', np.nan)))
        else:
            features.append(row.get('Rush1' + p, row.get(p.strip('_'), np.nan)))

    # Compute distances to all defender indices and pick 5 closest
    distances = []
    # Use canonical X/Y columns as the rusher base coordinates for distance computation
    rush_x = row.get('X', row.get('Rush1_X', np.nan))
    rush_y = row.get('Y', row.get('Rush1_Y', np.nan))
    for idx in def_indices:
        dx = row.get(f'Def{idx}_X', np.nan)
        dy = row.get(f'Def{idx}_Y', np.nan)
        try:
            dist = np.sqrt((rush_x - dx) ** 2 + (rush_y - dy) ** 2)
        except Exception:
            dist = np.nan
        distances.append((idx, dist))

    # Filter out defenders with NaN distances and sort
    distances = [d for d in distances if not pd.isna(d[1])]
    distances.sort(key=lambda x: x[1])
    closest = [idx for idx, _ in distances[:5]]

    # For each closest-defender slot, append defender metrics and relative metrics
    for slot in range(5):
        if slot < len(closest):
            didx = closest[slot]
            for p in pos_cols:
                # p includes leading underscore; use Def{didx}{p} to match column names like 'Def1_X'
                features.append(row.get(f'Def{didx}{p}', np.nan))
            # relative metrics
            dx = row.get(f'Def{didx}_X', np.nan)
            dy = row.get(f'Def{didx}_Y', np.nan)
            ds = row.get(f'Def{didx}_S', np.nan)
            da = row.get(f'Def{didx}_A', np.nan)
            # distances and diffs
            features.append(abs(rush_x - dx) if not pd.isna(rush_x) and not pd.isna(dx) else np.nan)
            features.append(abs(rush_y - dy) if not pd.isna(rush_y) and not pd.isna(dy) else np.nan)
            features.append((row.get('Rush1_S', np.nan) - ds) if not pd.isna(row.get('Rush1_S', np.nan)) and not pd.isna(ds) else np.nan)
            features.append((row.get('Rush1_A', np.nan) - da) if not pd.isna(row.get('Rush1_A', np.nan)) and not pd.isna(da) else np.nan)
        else:
            # pad with NaNs when fewer than 5 defenders present
            for _ in pos_cols:
                features.append(np.nan)
            for _ in range(4):
                features.append(np.nan)

    feature_data.append(features)

# Convert to numpy array
X = np.array(feature_data)
y = condensed_train['Yards'].values

# Create preprocessing and model pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Changed to median for better robustness
    ('scaler', StandardScaler()),
    ('regressor', GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=3,
        subsample=0.8,
        random_state=42
    ))
])

# Split the data into training and testing sets
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline and make predictions
print("Training model...")
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Print model performance metrics
print('\nModel Performance Metrics:')
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test, y_pred), 2))
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test, y_pred), 2))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 2))
print('R2 Score:', round(metrics.r2_score(y_test, y_pred), 3))

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Yards')
plt.ylabel('Predicted Yards')
plt.title('Actual vs Predicted Yards Gained')
plt.tight_layout()
plt.savefig('yards_prediction_results.png')
plt.close()

# Build accurate feature names that match the order used to construct X
all_feature_names = []
# play features
all_feature_names.extend(play_features)
# rusher features (keep Rush1_<p> naming for consistency)
for p in pos_cols:
    all_feature_names.append('Rush1' + p)
# closest defender slots: defender raw metrics then relative metrics
for slot in range(1, 6):
    for p in pos_cols:
        all_feature_names.append(f'Closest_Def{slot}_{p.strip("_")}')
    for rel in ['X_Dist', 'Y_Dist', 'S_Diff', 'A_Diff']:
        all_feature_names.append(f'Rush_ClosestDef{slot}_{rel}')

# Get feature importance
feature_importance = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': pipeline.named_steps['regressor'].feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 6))
sns.barplot(data=feature_importance.head(15), x='Importance', y='Feature')
plt.title('Top 15 Most Important Features')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

print('\nTop 15 Most Important Features:')
print(feature_importance.head(15).to_string(index=False))

# Plot residuals
plt.figure(figsize=(10, 6))
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Yards')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.tight_layout()
plt.savefig('residuals.png')
plt.close()

# Additional analysis: Error distribution
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=50, edgecolor='black')
plt.xlabel('Prediction Error (yards)')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Errors')
plt.tight_layout()
plt.savefig('error_distribution.png')
plt.close()

# Print error statistics
print('\nError Distribution Statistics:')
print('Mean Error:', round(np.mean(residuals), 2))
print('Median Error:', round(np.median(residuals), 2))
print('Standard Deviation of Error:', round(np.std(residuals), 2))
print('25th Percentile Error:', round(np.percentile(residuals, 25), 2))
print('75th Percentile Error:', round(np.percentile(residuals, 75), 2))
