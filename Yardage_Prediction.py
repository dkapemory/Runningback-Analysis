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
condensed_train = pd.read_csv('condensed_train_clean.csv', low_memory=False)

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

# Prepare features and target (use cleaned condensed CSV schema)
print("Preparing features from cleaned CSV schema...")

# Define rusher features present in the cleaned CSV
rusher_features = ['Rusher_X', 'Rusher_Y', 'Rusher_S', 'Rusher_A', 'Rusher_Dis', 'Rusher_Dir', 'Rusher_Weight', 'Rusher_Height']

# Define defender feature groups (numeric-only, no raw X/Y coordinates)
def_raw = ['PlayerWeight', 'PlayerHeight', 'S', 'A', 'Dis', 'Dir']
def_rel = ['Rush_ClosestDef{}_Dist', 'Rush_ClosestDef{}_S_Diff', 'Rush_ClosestDef{}_A_Diff']

# Build the ordered list of feature column names we will extract
all_feature_names = []
all_feature_names.extend(play_features)
all_feature_names.extend(rusher_features)
for slot in range(1, 6):
    for attr in def_raw:
        all_feature_names.append(f'Closest_Def{slot}_{attr}')
    for rel_tpl in def_rel:
        all_feature_names.append(rel_tpl.format(slot))

feature_data = []
# Only keep features that actually exist in the cleaned CSV. Do this silently
# so we don't complain about user-removed columns.
all_feature_names = [c for c in all_feature_names if c in condensed_train.columns]

for _, row in condensed_train.iterrows():
    features = [row.get(c, np.nan) for c in all_feature_names]
    feature_data.append(features)

# Convert feature_data to a DataFrame so we can drop any all-NaN columns
X_df = pd.DataFrame(feature_data, columns=all_feature_names)
# Identify and drop columns that are entirely NaN (these trigger imputer warnings)
cols_dropped = [c for c in X_df.columns if X_df[c].isna().all()]
if cols_dropped:
    print(f"Dropping {len(cols_dropped)} all-NaN feature columns:", cols_dropped)
    X_df = X_df.drop(columns=cols_dropped)

# Final feature names and matrix
final_feature_names = X_df.columns.tolist()
X = X_df.values
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

# Get feature importance (use names built from cleaned CSV schema)
fi = pipeline.named_steps['regressor'].feature_importances_
# Use the final_feature_names (after dropping empty cols) when available
if 'final_feature_names' in globals() or 'final_feature_names' in locals():
    name_list = final_feature_names
else:
    name_list = all_feature_names

n_imp = fi.shape[0]
n_names = len(name_list)
if n_imp != n_names:
    print(f'Warning: feature name count ({n_names}) != model feature count ({n_imp}). Adjusting names to match.')
    if n_names < n_imp:
        for i in range(n_names, n_imp):
            name_list.append(f'feature_{i}')
    else:
        name_list = name_list[:n_imp]

feature_importance = pd.DataFrame({
    'Feature': name_list,
    'Importance': fi
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
