import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

# Define paths
TRAIN_PATH = "dataset/train.csv"

# ==============================================================================
# Helper Functions (Custom and modular to run all 3 variations in one go)
# ==============================================================================

def add_time_features(df):
    dt = pd.to_datetime(df["timestamp"], format="%H:%M")
    df["hour"] = dt.dt.hour
    df["minute"] = dt.dt.minute
    df["time_slot"] = df["hour"] * 4 + df["minute"] // 15
    df["time_slot_sin"] = np.sin(2 * np.pi * df["time_slot"] / 96)
    df["time_slot_cos"] = np.cos(2 * np.pi * df["time_slot"] / 96)
    df["day_of_week"] = df["day"] % 7
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_peak_hour"] = (((df["hour"] >= 7) & (df["hour"] <= 10)) | ((df["hour"] >= 17) & (df["hour"] <= 20))).astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] < 6)).astype(int)
    df["is_office_hour"] = ((df["hour"] >= 9) & (df["hour"] < 17) & (df["is_weekend"] == 0)).astype(int)
    return df

def clean_and_prep(raw_train):
    # Split
    train_mask = (raw_train["day"] == 48) | ((raw_train["day"] == 49) & (raw_train["timestamp"].isin(["0:0", "0:15", "0:30"])))
    holdout_mask = (raw_train["day"] == 49) & (raw_train["timestamp"].isin(["0:45", "1:0", "1:15", "1:30", "1:45", "2:0"]))
    
    train_df = raw_train[train_mask].copy()
    holdout_df = raw_train[holdout_mask].copy()
    
    train_df["geohash_raw"] = train_df["geohash"].astype(str)
    holdout_df["geohash_raw"] = holdout_df["geohash"].astype(str)
    
    train_df = add_time_features(train_df)
    holdout_df = add_time_features(holdout_df)
    
    # Decoded Lat/Lon coordinates (hardcoded simple map or mean fallback to stay self-contained)
    import pygeohash as pgh
    geohashes = pd.concat([train_df["geohash_raw"], holdout_df["geohash_raw"]]).unique()
    geo_coords = {g: pgh.decode(g) for g in geohashes}
    
    for df in (train_df, holdout_df):
        df["lat"] = df["geohash_raw"].map(lambda g: geo_coords[g][0])
        df["lon"] = df["geohash_raw"].map(lambda g: geo_coords[g][1])
        df["lat"] = df["lat"].fillna(train_df["lat"].mean() if "lat" in train_df else 0.0)
        df["lon"] = df["lon"].fillna(train_df["lon"].mean() if "lon" in train_df else 0.0)
        
    # Spatial clusters
    kmeans = KMeans(n_clusters=10, random_state=42, n_init="auto")
    train_df["spatial_cluster"] = kmeans.fit_predict(train_df[["lat", "lon"]])
    holdout_df["spatial_cluster"] = kmeans.predict(holdout_df[["lat", "lon"]])
    
    # Categories imputation
    train_df["Weather"] = train_df["Weather"].fillna("Unknown")
    holdout_df["Weather"] = holdout_df["Weather"].fillna("Unknown")
    train_df["RoadType"] = train_df["RoadType"].fillna("Unknown")
    holdout_df["RoadType"] = holdout_df["RoadType"].fillna("Unknown")
    
    temp_median = train_df["Temperature"].median()
    geo_temp = train_df.groupby("geohash_raw")["Temperature"].median().to_dict()
    for df in (train_df, holdout_df):
        df["Temperature"] = df.apply(
            lambda r: geo_temp.get(r["geohash_raw"], temp_median) if pd.isna(r["Temperature"]) else r["Temperature"],
            axis=1
        )
        df["geo_4"] = df["geohash_raw"].str[:4]
        df["geo_5"] = df["geohash_raw"].str[:5]
        df["road_hour"] = df["RoadType"].astype(str) + "_" + df["hour"].astype(str)
        df["weather_hour"] = df["Weather"].astype(str) + "_" + df["hour"].astype(str)
        df["lane_hour"] = df["NumberofLanes"].astype(str) + "_" + df["hour"].astype(str)
        df["geo_hour"] = df["geohash_raw"] + "_" + df["hour"].astype(str)
        
    return train_df, holdout_df

# ==============================================================================
# Model Training & Ensembling
# ==============================================================================

def run_training(train_df, features):
    x_train = train_df[features]
    y_train = train_df["demand"]
    
    models = [
        LGBMRegressor(n_estimators=1000, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1),
        XGBRegressor(n_estimators=1000, learning_rate=0.03, max_depth=5, random_state=42, verbosity=0, n_jobs=1),
        CatBoostRegressor(iterations=1000, learning_rate=0.03, depth=5, random_state=42, allow_writing_files=False, verbose=0)
    ]
    for model in models:
        model.fit(x_train, y_train)
    return models

def predict_ensemble(models, x_predict):
    weights = np.array([0.15, 0.45, 0.40])
    predictions = np.column_stack([model.predict(x_predict) for model in models])
    return np.clip(predictions @ weights, 0.0, 1.0)

# ==============================================================================
# MAIN SCRIPT EXECUTION
# ==============================================================================

print("Loading raw dataset...")
raw_train = pd.read_csv(TRAIN_PATH)
train_df, holdout_df = clean_and_prep(raw_train)

# Static Day 48 aggregates setup
train_48 = train_df[train_df["day"] == 48].copy()
slot_mean = train_48.groupby("time_slot")["demand"].mean().to_dict()
geo_mean = train_48.groupby("geohash_raw")["demand"].mean().to_dict()
geo_std = train_48.groupby("geohash_raw")["demand"].std().to_dict()
geo_hour_mean = train_48.groupby("geo_hour")["demand"].mean().to_dict()
hour_mean = train_48.groupby("hour")["demand"].mean().to_dict()
weather_mean = train_48.groupby("Weather")["demand"].mean().to_dict()

# Add static stats to both
for df in (train_df, holdout_df):
    fallback = df["time_slot"].map(slot_mean)
    df["geo_mean_d48"] = df["geohash_raw"].map(geo_mean).fillna(fallback)
    df["geo_std_d48"] = df["geohash_raw"].map(geo_std).fillna(0.0)
    df["geo_hour_mean_d48"] = df["geo_hour"].map(geo_hour_mean).fillna(df["geo_mean_d48"])
    df["hour_mean_d48"] = df["hour"].map(hour_mean).fillna(fallback)
    df["weather_mean_d48"] = df["Weather"].map(weather_mean).fillna(fallback)
    df["slot_mean_d48"] = fallback

# Frequency Encodings
freq_base = train_df.copy()
for col in ["geohash_raw", "geo_hour", "road_hour"]:
    counts = freq_base[col].value_counts().to_dict()
    train_df[f"{col}_freq"] = train_df[col].map(counts).fillna(0.0)
    holdout_df[f"{col}_freq"] = holdout_df[col].map(counts).fillna(0.0)

# Build history maps for same-day lags
history_maps_train = {
    48: train_df[train_df["day"] == 48].set_index(["geohash_raw", "time_slot"])["demand"].to_dict(),
    49: train_df[train_df["day"] == 49].set_index(["geohash_raw", "time_slot"])["demand"].to_dict(),
}

# Add Day 48 individual static anchors (prevent target leakage on Day 48 rows)
d48_demand_dict = train_48.set_index(["geohash_raw", "time_slot"])["demand"].to_dict()
for df in (train_df, holdout_df):
    fallback = df["time_slot"].map(slot_mean)
    df["demand_prev_day"] = df.apply(lambda r: d48_demand_dict.get((r["geohash_raw"], r["time_slot"]), fallback.get(r["time_slot"])) if r["day"] == 49 else fallback.get(r["time_slot"]), axis=1)
    df["lag_1_d48"] = df.apply(lambda r: d48_demand_dict.get((r["geohash_raw"], r["time_slot"] - 1), fallback.get(r["time_slot"] - 1)) if r["day"] == 49 else fallback.get(r["time_slot"] - 1), axis=1)
    df["lag_2_d48"] = df.apply(lambda r: d48_demand_dict.get((r["geohash_raw"], r["time_slot"] - 2), fallback.get(r["time_slot"] - 2)) if r["day"] == 49 else fallback.get(r["time_slot"] - 2), axis=1)
    df["lag_3_d48"] = df.apply(lambda r: d48_demand_dict.get((r["geohash_raw"], r["time_slot"] - 3), fallback.get(r["time_slot"] - 3)) if r["day"] == 49 else fallback.get(r["time_slot"] - 3), axis=1)

# Categorical encodings
cat_cols = ["Weather", "RoadType", "LargeVehicles", "Landmarks", "geo_4", "geo_5", "road_hour", "weather_hour", "lane_hour"]
for col in cat_cols:
    encoder = LabelEncoder()
    combined = pd.concat([train_df[col].astype(str), holdout_df[col].astype(str)])
    encoder.fit(combined)
    train_df[col] = encoder.transform(train_df[col].astype(str))
    holdout_df[col] = encoder.transform(holdout_df[col].astype(str))

# Create same-day lags for training split (always uses ground truth)
def get_same_day_lags(df, history_maps, slot_mean):
    day = int(df["day"].iloc[0])
    history = history_maps.get(day, {})
    fallback = df["time_slot"].map(slot_mean)
    for lag in (1, 2, 3):
        df[f"same_day_lag_{lag}"] = [
            history.get((geohash, slot - lag), np.nan)
            for geohash, slot in zip(df["geohash_raw"], df["time_slot"])
        ]
        df[f"same_day_lag_{lag}"] = df[f"same_day_lag_{lag}"].fillna(fallback)
    df["rolling_mean_3_same"] = (df["same_day_lag_1"] + df["same_day_lag_2"] + df["same_day_lag_3"]) / 3.0
    df["diff_1_same"] = df["same_day_lag_1"] - df["same_day_lag_2"]
    return df

train_features_df = pd.concat(
    [get_same_day_lags(part.copy(), history_maps_train, slot_mean) for _, part in train_df.groupby("day", sort=True)],
    ignore_index=True
)

# Base Feature set (Mode 1 & Mode 2)
base_features = [
    "geo_4", "geo_5", "time_slot_sin", "time_slot_cos", "lat", "lon", "RoadType", 
    "NumberofLanes", "LargeVehicles", "Landmarks", "Temperature", "Weather",
    "same_day_lag_1", "same_day_lag_2", "same_day_lag_3", "rolling_mean_3_same", "diff_1_same",
    "geo_mean_d48", "geo_std_d48", "geo_hour_mean_d48", "hour_mean_d48", "weather_mean_d48", "slot_mean_d48",
    "road_hour", "weather_hour", "lane_hour", "geohash_raw_freq", "geo_hour_freq", "road_hour_freq",
    "is_weekend", "is_peak_hour", "is_night", "is_office_hour", "spatial_cluster"
]

# Anchored Feature set (Mode 3)
anchored_features = base_features + ["demand_prev_day", "lag_1_d48", "lag_2_d48", "lag_3_d48"]

# ==============================================================================
# MODE 1: FUTURE PROXY WITHOUT RECURSIVE FEED (PERFECT FORESIGHT SAME-DAY LAGS)
# ==============================================================================
print("\n--- RUNNING MODE 1: Perfect foresight same-day lags (without recursive feed) ---")
models_m1 = run_training(train_features_df, base_features)

# Use true same-day lags from the dataset for evaluation
combined_all = pd.concat([train_df, holdout_df], ignore_index=True)
history_maps_all = {
    48: combined_all[combined_all["day"] == 48].set_index(["geohash_raw", "time_slot"])["demand"].to_dict(),
    49: combined_all[combined_all["day"] == 49].set_index(["geohash_raw", "time_slot"])["demand"].to_dict(),
}
holdout_m1 = holdout_df.copy()
# Map raw geohash same-day lags directly from raw_train
fallback_h = holdout_m1["time_slot"].map(slot_mean)
for lag in (1, 2, 3):
    holdout_m1[f"same_day_lag_{lag}"] = [
        history_maps_all[49].get((geohash, slot - lag), fallback_h.iloc[i])
        for i, (geohash, slot) in enumerate(zip(holdout_m1["geohash_raw"], holdout_m1["time_slot"]))
    ]
holdout_m1["rolling_mean_3_same"] = (holdout_m1["same_day_lag_1"] + holdout_m1["same_day_lag_2"] + holdout_m1["same_day_lag_3"]) / 3.0
holdout_m1["diff_1_same"] = holdout_m1["same_day_lag_1"] - holdout_m1["same_day_lag_2"]

preds_m1 = predict_ensemble(models_m1, holdout_m1[base_features])
r2_m1 = r2_score(holdout_m1["demand"], preds_m1)
print(f"Mode 1 R2 Score: {r2_m1*100:.2f}% (R2: {r2_m1:.5f})")

# ==============================================================================
# MODE 2: FUTURE PROXY WITH RECURSIVE PREDICTION FEED
# ==============================================================================
print("\n--- RUNNING MODE 2: Recursive same-day lags (predicted lags fed recursively) ---")
models_m2 = run_training(train_features_df, base_features)

d49_history_m2 = train_features_df[train_features_df["day"] == 49].set_index(
    ["geohash_raw", "time_slot"]
)["demand"].to_dict()

ordered_holdout = holdout_df.sort_values(["time_slot", "Index"])
preds_m2_list = []
idx_m2_list = []

for _, slot_frame in ordered_holdout.groupby("time_slot", sort=True):
    batch = slot_frame.copy()
    slot = int(batch["time_slot"].iloc[0])
    fallback = slot_mean.get(slot, train_features_df["demand"].mean())
    
    for lag in (1, 2, 3):
        batch[f"same_day_lag_{lag}"] = [
            d49_history_m2.get((geohash, slot - lag), fallback)
            for geohash in batch["geohash_raw"]
        ]
    batch["rolling_mean_3_same"] = (batch["same_day_lag_1"] + batch["same_day_lag_2"] + batch["same_day_lag_3"]) / 3.0
    batch["diff_1_same"] = batch["same_day_lag_1"] - batch["same_day_lag_2"]
    
    preds = predict_ensemble(models_m2, batch[base_features])
    for index_val, geohash, pred in zip(batch["Index"], batch["geohash_raw"], preds):
        d49_history_m2[(geohash, slot)] = float(pred)
        preds_m2_list.append(float(pred))
        idx_m2_list.append(int(index_val))

eval_m2 = pd.DataFrame({"Index": idx_m2_list, "pred_demand": preds_m2_list}).set_index("Index")
eval_m2 = holdout_df.set_index("Index").join(eval_m2)
r2_m2 = r2_score(eval_m2["demand"], eval_m2["pred_demand"])
print(f"Mode 2 R2 Score: {r2_m2*100:.2f}% (R2: {r2_m2:.5f})")

# ==============================================================================
# MODE 3: FUTURE PROXY WITH RECURSIVE FEED + DAY 48 ANCHOR
# ==============================================================================
print("\n--- RUNNING MODE 3: Recursive same-day lags + Day 48 Anchor (static historical features) ---")
models_m3 = run_training(train_features_df, anchored_features)

d49_history_m3 = train_features_df[train_features_df["day"] == 49].set_index(
    ["geohash_raw", "time_slot"]
)["demand"].to_dict()

preds_m3_list = []
idx_m3_list = []

for _, slot_frame in ordered_holdout.groupby("time_slot", sort=True):
    batch = slot_frame.copy()
    slot = int(batch["time_slot"].iloc[0])
    fallback = slot_mean.get(slot, train_features_df["demand"].mean())
    
    for lag in (1, 2, 3):
        batch[f"same_day_lag_{lag}"] = [
            d49_history_m3.get((geohash, slot - lag), fallback)
            for geohash in batch["geohash_raw"]
        ]
    batch["rolling_mean_3_same"] = (batch["same_day_lag_1"] + batch["same_day_lag_2"] + batch["same_day_lag_3"]) / 3.0
    batch["diff_1_same"] = batch["same_day_lag_1"] - batch["same_day_lag_2"]
    
    preds = predict_ensemble(models_m3, batch[anchored_features])
    for index_val, geohash, pred in zip(batch["Index"], batch["geohash_raw"], preds):
        d49_history_m3[(geohash, slot)] = float(pred)
        preds_m3_list.append(float(pred))
        idx_m3_list.append(int(index_val))

eval_m3 = pd.DataFrame({"Index": idx_m3_list, "pred_demand": preds_m3_list}).set_index("Index")
eval_m3 = holdout_df.set_index("Index").join(eval_m3)
r2_m3 = r2_score(eval_m3["demand"], eval_m3["pred_demand"])
print(f"Mode 3 R2 Score: {r2_m3*100:.2f}% (R2: {r2_m3:.5f})")

# ==============================================================================
# FINAL SUMMARY COMPARISON
# ==============================================================================
print("\n" + "="*70)
print("                    EXPERIMENT RESULTS & EXPECTED SCORES")
print("="*70)
print(f"1. Mode 1 (True Lags / No Recursive):    {r2_m1*100:.3f}%  (R2: {r2_m1:.5f})")
print(f"2. Mode 2 (Pure Recursive Feed):        {r2_m2*100:.3f}%  (R2: {r2_m2:.5f})")
print(f"3. Mode 3 (Recursive Feed + D48 Anchor): {r2_m3*100:.3f}%  (R2: {r2_m3:.5f})")
print("-"*70)

# Estimate based on baseline 89.0 leaderboard score
print(f"Estimated Leaderboard Score for Mode 2 (Current V3): {89.0 + (r2_m2 - 0.9252)*100:.2f}")
print(f"Estimated Leaderboard Score for Mode 3 (Anchored):   {89.0 + (r2_m3 - 0.9252)*100:.2f}")
print("="*70)
