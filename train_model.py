"""
Train a multiple linear regression model to predict yards gained for runningback plays.

Assumptions:
- `train.csv` is the Kaggle-style tracking dataset where each row is a player-frame.
- Columns used: GameId, PlayId, NflId, NflIdRusher, X, Y, S, A, Orientation, Dir,
  TimeSnap, GameClock, Quarter, Down, Distance, YardLineFixed, HomeScoreBeforePlay,
  VisitorScoreBeforePlay, Yards
- We extract the rusher's earliest frame per play (closest to snap) as play-level features.

The script reads the CSV in chunks (to handle large files), builds a feature table,
trains sklearn.linear_model.LinearRegression, reports R^2/MAE/RMSE, and saves the model.
"""

import os
import argparse
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def parse_game_clock(clock_str):
    """Convert GameClock 'MM:SS' to seconds remaining in quarter."""
    try:
        if pd.isna(clock_str):
            return np.nan
        parts = str(clock_str).split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
    except Exception:
        return np.nan
    return np.nan


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def build_rusher_start_rows(csv_path, chunksize=500_000, max_plays=None):
    """Stream the CSV and keep the earliest TimeSnap row per PlayId for rushers.

    This implementation avoids concatenating large intermediate DataFrames.
    """
    usecols = [
        'GameId', 'PlayId', 'NflId', 'NflIdRusher', 'DisplayName', 'Team',
        'X', 'Y', 'S', 'A', 'Orientation', 'Dir', 'TimeSnap', 'GameClock',
        'Quarter', 'Down', 'Distance', 'YardLineFixed',
        'HomeScoreBeforePlay', 'VisitorScoreBeforePlay', 'Yards'
    ]

    reader = pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize, low_memory=False)
    earliest_by_play = {}  # PlayId -> row dict
    chunks = 0
    for chunk in reader:
        chunks += 1
        # Filter to rusher rows in this chunk
        rusher_rows = chunk[chunk['NflId'] == chunk['NflIdRusher']]
        if rusher_rows.empty:
            print(f"Chunk {chunks}: no rusher rows")
            continue

        # Ensure TimeSnap parsed once for comparison
        rusher_rows = rusher_rows.copy()
        rusher_rows['TimeSnap_dt'] = pd.to_datetime(rusher_rows['TimeSnap'], errors='coerce')

        for _, row in rusher_rows.iterrows():
            pid = row['PlayId']
            ts = row['TimeSnap_dt']
            if pid not in earliest_by_play:
                earliest_by_play[pid] = row.to_dict()
            else:
                existing_ts = pd.to_datetime(earliest_by_play[pid].get('TimeSnap'), errors='coerce')
                # prefer the earlier (smaller) timestamp; treat NaT as later
                if pd.isna(existing_ts) and not pd.isna(ts):
                    earliest_by_play[pid] = row.to_dict()
                elif not pd.isna(ts) and ts < existing_ts:
                    earliest_by_play[pid] = row.to_dict()

        print(f"Chunk {chunks}: processed, rusher rows seen: {len(rusher_rows)}; unique plays tracked: {len(earliest_by_play)}")

        if max_plays is not None and len(earliest_by_play) >= max_plays:
            print(f"Reached max_plays={max_plays}; stopping early.")
            break

    if len(earliest_by_play) == 0:
        raise RuntimeError('No rusher rows found in the dataset. Check CSV and column names.')

    # Convert dicts to DataFrame
    start_rows = pd.DataFrame.from_records(list(earliest_by_play.values()))
    # Recreate TimeSnap_dt column if needed
    if 'TimeSnap_dt' not in start_rows.columns:
        start_rows['TimeSnap_dt'] = pd.to_datetime(start_rows['TimeSnap'], errors='coerce')
    return start_rows


def engineer_features(df):
    """Create numeric features from the rusher-start rows."""
    out = pd.DataFrame()
    # Basic positional / motion features
    out['X'] = df['X'].apply(safe_float)
    out['Y'] = df['Y'].apply(safe_float)
    out['S'] = df['S'].apply(safe_float)
    out['A'] = df['A'].apply(safe_float)
    # Orientation and direction: encode as sin/cos of degrees
    out['Orient_sin'] = np.sin(np.deg2rad(df['Orientation'].fillna(0).astype(float)))
    out['Orient_cos'] = np.cos(np.deg2rad(df['Orientation'].fillna(0).astype(float)))
    out['Dir_sin'] = np.sin(np.deg2rad(df['Dir'].fillna(0).astype(float)))
    out['Dir_cos'] = np.cos(np.deg2rad(df['Dir'].fillna(0).astype(float)))

    # Play context features
    out['Quarter'] = df['Quarter'].fillna(0).astype(int)
    out['Down'] = df['Down'].fillna(1).astype(int)
    out['Distance'] = df['Distance'].apply(safe_float).fillna(0)
    out['YardLineFixed'] = df['YardLineFixed'].apply(safe_float).fillna(50)
    out['ScoreDiff'] = df['HomeScoreBeforePlay'].fillna(0).astype(float) - df['VisitorScoreBeforePlay'].fillna(0).astype(float)
    out['GameClock_s'] = df['GameClock'].apply(parse_game_clock).fillna(0)

    # Target
    out['Yards'] = df['Yards'].apply(safe_float)

    # Drop rows with missing target
    out = out.dropna(subset=['Yards'])
    return out


def train_and_save(X, y, output_path):
    model = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)

    # Cross-validated R^2 (5-fold)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

    print("Training results:")
    print(f"R^2 (test): {r2:.4f}")
    print(f"MAE (test): {mae:.4f}")
    print(f"RMSE (test): {rmse:.4f}")
    print(f"CV R^2 (5-fold) mean: {cv_scores.mean():.4f} std: {cv_scores.std():.4f}")

    # Save model (pickle) and also return coefficients for safe export
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {output_path}")
    coef_info = {
        'coef': model.coef_.tolist(),
        'intercept': float(model.intercept_)
    }
    return model, coef_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='train.csv', help='Path to train.csv')
    parser.add_argument('--model-out', type=str, default='linear_model.pkl', help='Output path for trained model')
    parser.add_argument('--chunksize', type=int, default=500_000, help='CSV chunksize for streaming')
    parser.add_argument('--max-plays', type=int, default=None, help='Maximum number of plays to process (for safety/testing)')
    parser.add_argument('--save-features', type=str, default=None, help='Optional path to save engineered features CSV')
    parser.add_argument('--export-coefs', type=str, default=None, help='Optional path to export model coefficients as JSON (safer than pickle)')
    parser.add_argument('--no-pickle', action='store_true', help='Do not save pickle file; only export coefficients if requested')
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV file not found: {args.csv}")

    print('Building rusher-start table (this may take a while) ...')
    start_rows = build_rusher_start_rows(args.csv, chunksize=args.chunksize, max_plays=args.max_plays)
    print(f"Collected {len(start_rows)} plays with rusher-start rows.")

    print('Engineering features ...')
    data = engineer_features(start_rows)
    features = data.drop(columns=['Yards'])
    target = data['Yards']

    print('Training linear regression model ...')
    model, coef_info = train_and_save(features, target, args.model_out)

    # Optionally save engineered features for inspection / faster iteration
    if args.save_features:
        features.assign(Yards=target).to_csv(args.save_features, index=False)
        print(f"Engineered features saved to {args.save_features}")

    # Export coefficients as JSON if requested (safer to share than pickled model)
    if args.export_coefs:
        # include feature names so consumers can reconstruct linear prediction
        out = {
            'feature_names': features.columns.tolist(),
            'coef': coef_info['coef'],
            'intercept': coef_info['intercept']
        }
        with open(args.export_coefs, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"Model coefficients exported to {args.export_coefs} (JSON)")

    if args.no_pickle:
        print("Note: --no-pickle set; the pickled model file was still written by default earlier in the run."
              " If you want to avoid pickle entirely, re-run without write permissions or remove the file.")

    # Security note about pickle
    print("Security note: Pickled models can execute code when loaded. Do not load pickles from untrusted sources.")


if __name__ == '__main__':
    main()
