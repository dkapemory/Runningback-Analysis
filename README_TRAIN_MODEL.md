Train a simple multiple linear regression model for runningback yards

This repository addition includes `train_model.py` which trains a linear regression
model to predict yards gained for runningback plays based on the rusher's starting
frame and play context.

Usage
-----
1. Install dependencies (prefer a virtualenv):

    pip3 install -r requirements.txt

2. Run the trainer (in the repository root):

    python train_model.py --csv train.csv --model-out linear_model.pkl

Notes & assumptions
-------------------
- The script reads `train.csv` in chunks and selects rows where `NflId == NflIdRusher`.
- For each PlayId it keeps the earliest TimeSnap row as the rusher's start-of-play data.
- It constructs a small set of numeric features (position, speed, orientation encodings,
  down, distance, yardline, quarter, score diff, and game clock seconds) and trains
  sklearn.linear_model.LinearRegression on the result.
- This is intentionally a simple baseline. Better models can aggregate defender
  distances, directions, or use the full tracking sequence.

Outputs
-------
- `linear_model.pkl` (by default) containing the trained sklearn LinearRegression model.
- Printed metrics: R^2, MAE, RMSE, and 5-fold CV R^2.
