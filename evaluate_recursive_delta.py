import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# Import functions and feature space from your main pipeline
from final_submission_v3 import (
    TRAIN_PATH,
    FEATURES,
    clean_columns,
    add_static_stats,
    build_history_maps,
    add_same_day_lags,
    encode_categories,
    train_models,
    blend_predictions
)

print("=== STARTING SEQUENTIAL RECURSIVE BACKTEST ===")
print("Loading train.csv...")
raw_train = pd.read_csv(TRAIN_PATH)

# We split the data to simulate a multi-step forecasting horizon on Day 49:
# - train_df: All of Day 48, and Day 49 slots 0, 1, 2 (up to 0:30 AM)
# - holdout_df: Day 49 slots 3, 4, 5, 6, 7, 8 (from 0:45 AM to 2:00 AM)

train_mask = (raw_train["day"] == 48) | (
    (raw_train["day"] == 49) & 
    (raw_train["timestamp"].isin(["0:0", "0:15", "0:30"]))
)
holdout_mask = (raw_train["day"] == 49) & (
    raw_train["timestamp"].isin(["0:45", "1:0", "1:15", "1:30", "1:45", "2:0"])
)

train_df = raw_train[train_mask].copy()
holdout_df = raw_train[holdout_mask].copy()

print(f"Training split rows: {len(train_df)}")
print(f"Holdout split rows:  {len(holdout_df)} (spanning 6 sequential time slots)")

# 1. Clean columns and process static coordinates/features
train_df, holdout_df = clean_columns(train_df, holdout_df)
train_df, holdout_df, slot_mean = add_static_stats(train_df, holdout_df)

# 2. Build history maps from the training set
history_maps = build_history_maps(train_df)

# 3. Add same-day lags for training data
train_df = pd.concat(
    [
        add_same_day_lags(part.copy(), history_maps, slot_mean)
        for _, part in train_df.groupby("day", sort=True)
    ],
    ignore_index=True,
)

# Encode categories
train_df, encoded_holdout = encode_categories(train_df.copy(), holdout_df.copy())
holdout_df = encoded_holdout

# 4. Train the ensemble models on the training split
print("\nTraining ensemble models (LightGBM, XGBoost, CatBoost)...")
models = train_models(train_df)

# 5. Perform recursive forecasting on holdout_df slot by slot
print("\nSimulating recursive forecasting slot-by-slot...")
d49_history = train_df[train_df["day"] == 49].set_index(
    ["geohash_raw", "time_slot"]
)["demand"].to_dict()

# We sort holdout by slot to predict sequentially
ordered_holdout = holdout_df.sort_values(["time_slot", "Index"])
output_predictions = []

for _, slot_frame in ordered_holdout.groupby("time_slot", sort=True):
    batch = slot_frame.copy()
    slot = int(batch["time_slot"].iloc[0])
    fallback = slot_mean.get(slot, train_df["demand"].mean())
    
    # Construct same-day lags using the current d49_history (which includes our previous predictions)
    for lag in (1, 2, 3):
        batch[f"same_day_lag_{lag}"] = [
            d49_history.get((geohash, slot - lag), fallback)
            for geohash in batch["geohash_raw"]
        ]
    
    batch["rolling_mean_3_same"] = (
        batch["same_day_lag_1"] + batch["same_day_lag_2"] + batch["same_day_lag_3"]
    ) / 3.0
    batch["diff_1_same"] = batch["same_day_lag_1"] - batch["same_day_lag_2"]
    
    # Predict demand using the trained ensemble models
    preds = blend_predictions(models, batch[FEATURES])
    
    # Store predictions in d49_history to be used as lag features for subsequent slots
    for index_val, geohash, pred in zip(batch["Index"], batch["geohash_raw"], preds):
        d49_history[(geohash, slot)] = float(pred)
        output_predictions.append((int(index_val), float(pred)))

# Match predictions back to the actual demands to evaluate R2 and RMSE
pred_df = pd.DataFrame(output_predictions, columns=["Index", "pred_demand"]).set_index("Index")
eval_df = holdout_df.set_index("Index").join(pred_df)

y_true = eval_df["demand"]
y_pred = eval_df["pred_demand"]

# Compute evaluation metrics
r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
accuracy = (1 - rmse) * 100

print("\n========================================================")
print("       RECURSIVE BACKTEST RESULTS (6-STEP HORIZON)     ")
print("========================================================")
print(f"R2 Score (Real Leaderboard Scale): {r2:.5f} ({r2*100:.2f}%)")
print(f"MAE:                                {mae:.5f}")
print(f"RMSE:                               {rmse:.5f}")
print(f"1-RMSE Accuracy Score:              {accuracy:.2f}%")
print("========================================================")
