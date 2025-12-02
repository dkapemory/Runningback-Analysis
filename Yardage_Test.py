

"""Yardage_Test.py

Use the trained model from Yardage_Prediction.py to predict yards gained
for new plays supplied in a CSV that has the same columns as
`condensed_train_clean.csv` **except** for the `Yards` column.

Usage (from the repo root):
    python Yardage_Test.py new_plays.csv

`new_plays.csv` should have one or more rows with the same feature columns
as `condensed_train_clean.csv` (no `Yards` column). The script will print
predictions to the terminal and also write `new_plays.predicted.csv` with a
`Predicted_Yards` column appended.

This script expects **either**:
  1. A saved model file `yardage_pipeline.joblib` created by
     Yardage_Prediction.py, **or**
  2. A `pipeline` object defined and trained at the top level of
     `Yardage_Prediction.py` so that importing it runs the training and
     exposes the fitted pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None  # type: ignore


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "yardage_pipeline.joblib"  # expected saved model file
TRAIN_CSV_PATH = HERE / "condensed_train_clean.csv"  # used only to get columns


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model():
    """Load the trained pipeline.

    Preference order:
      1. Load a joblib file at MODEL_PATH if it exists.
      2. Import `pipeline` from Yardage_Prediction.py (which should train
         and expose a fitted pipeline at module import time).
    """

    # Option 1: load from joblib if available
    if MODEL_PATH.exists() and joblib is not None:
        print(f"Loading trained model from {MODEL_PATH}...")
        return joblib.load(MODEL_PATH)

    # Option 2: fall back to importing the pipeline from Yardage_Prediction
    print("Joblib model file not found; importing pipeline from Yardage_Prediction.py...")
    try:
        from Yardage_Prediction import pipeline  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Could not load a trained model. Either save the fitted pipeline "
            "to 'yardage_pipeline.joblib' in Yardage_Prediction.py, or expose "
            "a fitted 'pipeline' object at the top level of that file."
        ) from exc

    return pipeline


# ---------------------------------------------------------------------------
# Feature handling
# ---------------------------------------------------------------------------

def get_feature_columns() -> Iterable[str]:
    """Return the list of feature column names used for training.

    This is inferred from `condensed_train_clean.csv` by taking all columns
    except `Yards`.
    """

    if not TRAIN_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {TRAIN_CSV_PATH}. Make sure this script is run from "
            f"the same directory where 'condensed_train_clean.csv' lives, or "
            f"update TRAIN_CSV_PATH in Yardage_Test.py."
        )

    sample = pd.read_csv(TRAIN_CSV_PATH, nrows=1, low_memory=False)
    feature_cols = [c for c in sample.columns if c != "Yards"]
    return feature_cols


def prepare_input_df(input_df: pd.DataFrame, feature_cols: Iterable[str]) -> pd.DataFrame:
    """Ensure the input dataframe matches the training feature space.

    * Adds any missing feature columns (filled with NaN so the pipeline's
      imputer can handle them).
    * Drops any extra columns the model never saw.
    * Reorders columns to match training order.
    """

    df = input_df.copy()

    # Add missing feature columns as NaN
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Drop any columns that were not used during training
    extra_cols = [c for c in df.columns if c not in feature_cols]
    if extra_cols:
        print(f"Dropping unused columns: {extra_cols}")
        df = df.drop(columns=extra_cols)

    # Reorder columns to match training
    df = df[list(feature_cols)]
    return df


# ---------------------------------------------------------------------------
# Public prediction helpers
# ---------------------------------------------------------------------------

def predict_from_csv(csv_path: Path) -> pd.DataFrame:
    """Load plays from a CSV and return a dataframe with predictions.

    The CSV should have the same feature columns as `condensed_train_clean.csv`
    (optionally including `Yards`, which will be ignored if present).
    """

    print(f"Loading plays from {csv_path}...")
    new_df = pd.read_csv(csv_path, low_memory=False)

    # Drop true yards if accidentally included
    if "Yards" in new_df.columns:
        new_df = new_df.drop(columns=["Yards"])

    feature_cols = list(get_feature_columns())
    X_new = prepare_input_df(new_df, feature_cols)

    model = load_model()
    print("Generating predictions...")
    preds = model.predict(X_new)

    result = new_df.copy()
    result["Predicted_Yards"] = preds
    return result


def predict_single_play(play_features: Dict[str, float]) -> float:
    """Predict yards gained for a single play represented as a dict.

    Example
    -------
    >>> from Yardage_Test import predict_single_play
    >>> play = {
    ...     "Quarter": 1,
    ...     "Down": 2,
    ...     "Distance": 5,
    ...     # ... all other feature columns from condensed_train_clean.csv ...
    ... }
    >>> predict_single_play(play)
    4.32
    """

    feature_cols = list(get_feature_columns())
    df = pd.DataFrame([play_features])
    X_new = prepare_input_df(df, feature_cols)
    model = load_model()
    pred = float(model.predict(X_new)[0])
    return pred


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        print(
            "Usage: python Yardage_Test.py path/to/new_plays.csv\n\n"
            "The CSV should have the same columns as 'condensed_train_clean.csv' "
            "except for the 'Yards' column (which will be ignored if present)."
        )
        return

    csv_path = Path(argv[0]).expanduser().resolve()
    if not csv_path.exists():
        print(f"Error: file not found: {csv_path}")
        return

    result = predict_from_csv(csv_path)

    # Print predictions to the console
    for idx, yards in enumerate(result["Predicted_Yards"].values):
        print(f"Row {idx}: predicted yards = {yards:.2f}")

    # Also save to a new CSV next to the input file
    out_path = csv_path.with_suffix(".predicted.csv")
    result.to_csv(out_path, index=False)
    print(f"\nSaved predictions with 'Predicted_Yards' to: {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()