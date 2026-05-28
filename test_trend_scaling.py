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

TRAIN_PATH = "dataset/train.csv"

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
    train_mask = (raw_train["day"] == 48) | ((raw_train["day"] == 49) & (raw_train["timestamp"].isin(["0:0", "0:15", "0:30"])))
    holdout_mask = (raw_train["day"] == 49) & (raw_train["timestamp"].isin(["0:45", "1:0", "1:15", "1:30", "1:45", "2:0"]))
    
    train_df = raw_train[train_mask].copy()
    holdout_df = raw_train[holdout_mask].copy()
    
    train_df["geohash_raw"] = train_df["geohash"].astype(str)
    holdout_df["geohash_raw"] = holdout_df["geohash"].astype(str)
    
    train_df = add_time_features(train_df)
    holdout_df = add_time_features(holdout_df)
    
    import pygeohash as pgh
    geohashes = pd.concat([train_df["geohash_raw"], holdout_df["geohash_raw"]]).unique()
    geo_coords = {g: pgh.decode(g) for g in geohashes}
    
    for df in (train_df, holdout_df):
        df["lat"] = df["geohash_raw"].map(lambda g: geo_coords[g][0])
        df["lon"] = df["geohash_raw"].map(lambda g: geo_coords[g][1])
        df["lat"] = df["lat"].fillna(train_df["lat"].mean() if "lat" in train_df else 0.0)
        df["lon"] = df["lon"].fillna(train_df["lon"].mean() if "lon" in train_df else 0.0)
        
    kmeans = KMeans(n_clusters=10, random_state=42, n_init="auto")
    train_df["spatial_cluster"] = kmeans.fit_predict(train_df[["lat", "lon"]])
    holdout_df["spatial_cluster"] = kmeans.predict(holdout_df[["lat", "lon"]])
    
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

print("Loading raw dataset...")
raw_train = pd.read_csv(TRAIN_PATH)
train_df, holdout_df = clean_and_prep(raw_train)

train_48 = train_df[train_df["day"] == 48].copy()
train_49 = train_df[train_df["day"] == 49].copy()

slot_mean = train_48.groupby("time_slot")["demand"].mean().to_dict()
geo_mean = train_48.groupby("geohash_raw")["demand"].mean().to_dict()
geo_std = train_48.groupby("geohash_raw")["demand"].std().to_dict()
geo_hour_mean = train_48.groupby("geo_hour")["demand"].mean().to_dict()
hour_mean = train_48.groupby("hour")["demand"].mean().to_dict()
weather_mean = train_48.groupby("Weather")["demand"].mean().to_dict()

for df in (train_df, holdout_df):
    fallback = df["time_slot"].map(slot_mean)
    df["geo_mean_d48"] = df["geohash_raw"].map(geo_mean).fillna(fallback)
    df["geo_std_d48"] = df["geohash_raw"].map(geo_std).fillna(0.0)
    df["geo_hour_mean_d48"] = df["geo_hour"].map(geo_hour_mean).fillna(df["geo_mean_d48"])
    df["hour_mean_d48"] = df["hour"].map(hour_mean).fillna(fallback)
    df["weather_mean_d48"] = df["Weather"].map(weather_mean).fillna(fallback)
    df["slot_mean_d48"] = fallback

freq_base = train_df.copy()
for col in ["geohash_raw", "geo_hour", "road_hour"]:
    counts = freq_base[col].value_counts().to_dict()
    train_df[f"{col}_freq"] = train_df[col].map(counts).fillna(0.0)
    holdout_df[f"{col}_freq"] = holdout_df[col].map(counts).fillna(0.0)

# ==============================================================================
# HIERARCHICAL DAY-TO-DAY TREND RATIOS CALCULATION
# ==============================================================================
print("\nCalculating location-specific day-to-day scaling factors...")

# Calculate geohash-specific mean demand for slots 0-2 (the starting slots of Day 49)
d48_start = train_48[train_48["time_slot"].isin([0, 1, 2])]
d49_start = train_49[train_49["time_slot"].isin([0, 1, 2])]

d48_start_mean = d48_start.groupby("geohash_raw")["demand"].mean().to_dict()
d49_start_mean = d49_start.groupby("geohash_raw")["demand"].mean().to_dict()

# Calculate global time slot means for fallback
global_d48_start = d48_start["demand"].mean()
global_d49_start = d49_start["demand"].mean()
global_ratio = global_d49_start / max(global_d48_start, 1e-6)

# Build a dictionary of location-specific day-to-day scaling ratios
location_ratios = {}
for g in geo_mean.keys():
    m48 = d48_start_mean.get(g, 0.0)
    m49 = d49_start_mean.get(g, 0.0)
    
    if m48 > 0.01:
        # Clip ratios between 0.5 and 2.0 to prevent outlier anomalies
        location_ratios[g] = np.clip(m49 / m48, 0.5, 2.0)
    else:
        location_ratios[g] = global_ratio

# Apply scaling ratios to Day 48 individual static anchors
d48_demand_dict = train_48.set_index(["geohash_raw", "time_slot"])["demand"].to_dict()

for df in (train_df, holdout_df):
    fallback = df["time_slot"].map(slot_mean)
    
    # We define a helper to lookup same-slot demand from Day 48 and scale it by the day-to-day trend ratio
    def get_scaled_anchor(row, offset=0):
        slot = row["time_slot"] + offset
        if row["day"] == 48:
            return slot_mean.get(slot, 0.0)
        
        g = row["geohash_raw"]
        ratio = location_ratios.get(g, global_ratio)
        
        val = d48_demand_dict.get((g, slot), slot_mean.get(slot, 0.0))
        if val is None or pd.isna(val):
            val = 0.0
        return np.clip(val * ratio, 0.0, 1.0)
        
    df["demand_prev_day"] = df.apply(lambda r: get_scaled_anchor(r, 0), axis=1)
    df["lag_1_d48"] = df.apply(lambda r: get_scaled_anchor(r, -1), axis=1)
    df["lag_2_d48"] = df.apply(lambda r: get_scaled_anchor(r, -2), axis=1)
    df["lag_3_d48"] = df.apply(lambda r: get_scaled_anchor(r, -3), axis=1)

cat_cols = ["Weather", "RoadType", "LargeVehicles", "Landmarks", "geo_4", "geo_5", "road_hour", "weather_hour", "lane_hour"]
for col in cat_cols:
    encoder = LabelEncoder()
    combined = pd.concat([train_df[col].astype(str), holdout_df[col].astype(str)])
    encoder.fit(combined)
    train_df[col] = encoder.transform(train_df[col].astype(str))
    holdout_df[col] = encoder.transform(holdout_df[col].astype(str))

# Same-day lags for training split (always uses ground truth)
history_maps_train = {
    48: train_df[train_df["day"] == 48].set_index(["geohash_raw", "time_slot"])["demand"].to_dict(),
    49: train_df[train_df["day"] == 49].set_index(["geohash_raw", "time_slot"])["demand"].to_dict(),
}

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

features = [
    "geo_4", "geo_5", "time_slot_sin", "time_slot_cos", "lat", "lon", "RoadType", 
    "NumberofLanes", "LargeVehicles", "Landmarks", "Temperature", "Weather",
    "same_day_lag_1", "same_day_lag_2", "same_day_lag_3", "rolling_mean_3_same", "diff_1_same",
    "geo_mean_d48", "geo_std_d48", "geo_hour_mean_d48", "hour_mean_d48", "weather_mean_d48", "slot_mean_d48",
    "road_hour", "weather_hour", "lane_hour", "geohash_raw_freq", "geo_hour_freq", "road_hour_freq",
    "is_weekend", "is_peak_hour", "is_night", "is_office_hour", "spatial_cluster",
    "demand_prev_day", "lag_1_d48", "lag_2_d48", "lag_3_d48"
]

print("Training Ensemble models on trend-scaled feature space...")
models = [
    LGBMRegressor(n_estimators=1000, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1),
    XGBRegressor(n_estimators=1000, learning_rate=0.03, max_depth=5, random_state=42, verbosity=0, n_jobs=1),
    CatBoostRegressor(iterations=1000, learning_rate=0.03, depth=5, random_state=42, allow_writing_files=False, verbose=0)
]
for model in models:
    model.fit(train_features_df[features], train_features_df["demand"])

# ==============================================================================
# HIERARCHICAL RECURSIVE FORECASTING WITH TREND-SCALING
# ==============================================================================
print("\nRunning Hierarchical Fallback Recursive Forecasting with Day-to-Day Trend-Scaling...")
d49_history = train_features_df[train_features_df["day"] == 49].set_index(
    ["geohash_raw", "time_slot"]
)["demand"].to_dict()

# Map Day 48 same-slot demand
d48_same_slot_dict = train_df[train_df["day"] == 48].set_index(
    ["geohash_raw", "time_slot"]
)["demand"].to_dict()

ordered_holdout = holdout_df.sort_values(["time_slot", "Index"])
preds_list = []
idx_list = []

for _, slot_frame in ordered_holdout.groupby("time_slot", sort=True):
    batch = slot_frame.copy()
    slot = int(batch["time_slot"].iloc[0])
    fallback = slot_mean.get(slot, train_features_df["demand"].mean())
    
    for lag in (1, 2, 3):
        # Look up recursive same-day, else look up trend-scaled Day 48, else fallback
        l_preds = []
        for geohash in batch["geohash_raw"]:
            if (geohash, slot - lag) in d49_history:
                val = d49_history[(geohash, slot - lag)]
            else:
                ratio = location_ratios.get(geohash, global_ratio)
                d48_val = d48_same_slot_dict.get((geohash, slot - lag), slot_mean.get(slot - lag, 0.0))
                if d48_val is None or pd.isna(d48_val):
                    d48_val = 0.0
                val = np.clip(d48_val * ratio, 0.0, 1.0)
            l_preds.append(val)
        batch[f"same_day_lag_{lag}"] = l_preds
        
    batch["rolling_mean_3_same"] = (batch["same_day_lag_1"] + batch["same_day_lag_2"] + batch["same_day_lag_3"]) / 3.0
    batch["diff_1_same"] = batch["same_day_lag_1"] - batch["same_day_lag_2"]
    
    # Predict
    predictions = np.column_stack([model.predict(batch[features]) for model in models])
    preds = np.clip(predictions @ np.array([0.15, 0.45, 0.40]), 0.0, 1.0)
    
    for index_val, geohash, pred in zip(batch["Index"], batch["geohash_raw"], preds):
        d49_history[(geohash, slot)] = float(pred)
        preds_list.append(float(pred))
        idx_list.append(int(index_val))

eval_df = pd.DataFrame({"Index": idx_list, "pred_demand": preds_list}).set_index("Index")
eval_df = holdout_df.set_index("Index").join(eval_df)
r2 = r2_score(eval_df["demand"], eval_df["pred_demand"])
print("\n========================================================")
print(f"Hierarchical Fallback + Trend Scaling R2 Score: {r2*100:.5f}% (R2: {r2:.5f})")
print("========================================================")
