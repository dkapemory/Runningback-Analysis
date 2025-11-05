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

# Load the condensed training data with low_memory=False to handle mixed types
print("Loading data...")
condensed_train = pd.read_csv('condensed_train.csv', low_memory=False)

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
feature_data = []
for _, row in condensed_train.iterrows():
    features = []
    # Add play features
    for col in play_features:
        features.append(row[col])
    
    # Add rusher features
    for col in ['Rush1' + p for p in pos_cols]:
        features.append(row[col])
    
    # Add defender features and calculate relative metrics
    for i in range(5):
        def_prefix = f'Def{i+1}'
        # Add basic defender metrics
        for p in pos_cols:
            features.append(row[def_prefix + p])
        
        # Add relative position metrics
        features.append(abs(row['Rush1_X'] - row[def_prefix + '_X']))  # X distance
        features.append(abs(row['Rush1_Y'] - row[def_prefix + '_Y']))  # Y distance
        features.append(row['Rush1_S'] - row[def_prefix + '_S'])       # Speed difference
        features.append(row['Rush1_A'] - row[def_prefix + '_A'])       # Acceleration difference
        
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

# Create feature names for importance plot
all_feature_names = (
    play_features +
    ['Rush1' + p for p in pos_cols] +
    [f'{def_prefix}{p}' for i in range(5) for def_prefix in [f'Def{i+1}_'] for p in pos_cols] +
    [f'Rush_Def{i+1}_{m}' for i in range(5) for m in ['X_Dist', 'Y_Dist', 'S_Diff', 'A_Diff']]
)

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
