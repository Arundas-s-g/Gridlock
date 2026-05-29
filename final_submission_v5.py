import warnings
import numpy as np
import pandas as pd
import pygeohash as pgh
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

TRAIN_PATH = "dataset/train.csv"
TEST_PATH = "dataset/test.csv"
SUBMISSION_PATH = "submission.csv"

# ==============================================================================
# 1. Feature Engineering & Preprocessing Functions
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
    df["is_peak_hour"] = (
        ((df["hour"] >= 7) & (df["hour"] <= 10))
        | ((df["hour"] >= 17) & (df["hour"] <= 20))
    ).astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] < 6)).astype(int)
    df["is_office_hour"] = (
        (df["hour"] >= 9) & (df["hour"] < 17) & (df["is_weekend"] == 0)
    ).astype(int)
    
    # New Temporal Engineered States
    df["part_of_day"] = 3
    df.loc[(df["hour"] >= 6) & (df["hour"] < 11), "part_of_day"] = 0
    df.loc[(df["hour"] >= 11) & (df["hour"] < 16), "part_of_day"] = 1
    df.loc[(df["hour"] >= 16) & (df["hour"] < 21), "part_of_day"] = 2
    
    df["rush_hour"] = 0
    df.loc[(df["hour"] >= 7) & (df["hour"] < 9), "rush_hour"] = 1
    df.loc[(df["hour"] >= 17) & (df["hour"] < 19), "rush_hour"] = 2
    
    return df


def add_geo_features(train_df, predict_df):
    geohashes = pd.concat([train_df["geohash_raw"], predict_df["geohash_raw"]]).unique()
    geo_coords = {}
    for geohash in geohashes:
        try:
            geo_coords[geohash] = pgh.decode(geohash)
        except Exception:
            geo_coords[geohash] = (np.nan, np.nan)

    for df in (train_df, predict_df):
        df["lat"] = df["geohash_raw"].map(lambda g: geo_coords[g][0])
        df["lon"] = df["geohash_raw"].map(lambda g: geo_coords[g][1])

    train_df["lat"] = train_df["lat"].fillna(train_df["lat"].mean())
    train_df["lon"] = train_df["lon"].fillna(train_df["lon"].mean())
    predict_df["lat"] = predict_df["lat"].fillna(train_df["lat"].mean())
    predict_df["lon"] = predict_df["lon"].fillna(train_df["lat"].mean())

    kmeans = KMeans(n_clusters=15, random_state=42, n_init="auto")
    train_df["spatial_cluster"] = kmeans.fit_predict(train_df[["lat", "lon"]])
    predict_df["spatial_cluster"] = kmeans.predict(predict_df[["lat", "lon"]])
    return train_df, predict_df


def clean_columns(train_df, predict_df):
    train_df["geohash_raw"] = train_df["geohash"].astype(str)
    predict_df["geohash_raw"] = predict_df["geohash"].astype(str)

    train_df = add_time_features(train_df)
    predict_df = add_time_features(predict_df)
    train_df, predict_df = add_geo_features(train_df, predict_df)

    train_df["Weather"] = train_df["Weather"].fillna("Unknown")
    predict_df["Weather"] = predict_df["Weather"].fillna("Unknown")
    train_df["RoadType"] = train_df["RoadType"].fillna("Unknown")
    predict_df["RoadType"] = predict_df["RoadType"].fillna("Unknown")

    temp_median = train_df["Temperature"].median()
    geo_temp = train_df.groupby("geohash_raw")["Temperature"].median().to_dict()
    for df in (train_df, predict_df):
        df["Temperature"] = df.apply(
            lambda row: geo_temp.get(row["geohash_raw"], temp_median)
            if pd.isna(row["Temperature"])
            else row["Temperature"],
            axis=1,
        )
        df["geo_4"] = df["geohash_raw"].str[:4]
        df["geo_5"] = df["geohash_raw"].str[:5]
        
        # Keys for interaction maps and encoding
        df["road_hour"] = df["RoadType"].astype(str) + "_" + df["hour"].astype(str)
        df["weather_hour"] = df["Weather"].astype(str) + "_" + df["hour"].astype(str)
        df["lane_hour"] = df["NumberofLanes"].astype(str) + "_" + df["hour"].astype(str)
        df["geo_hour"] = df["geohash_raw"] + "_" + df["hour"].astype(str)
        df["geo_time_slot"] = df["geohash_raw"] + "_" + df["time_slot"].astype(str)
        df["road_slot"] = df["RoadType"].astype(str) + "_" + df["time_slot"].astype(str)
        df["weather_slot"] = df["Weather"].astype(str) + "_" + df["time_slot"].astype(str)

    return train_df, predict_df


def add_static_stats(train_df, predict_df):
    train_48 = train_df[train_df["day"] == 48].copy()
    global_mean = train_48["demand"].mean()
    slot_mean = train_48.groupby("time_slot")["demand"].mean().to_dict()
    geo_mean = train_48.groupby("geohash_raw")["demand"].mean().to_dict()
    geo_std = train_48.groupby("geohash_raw")["demand"].std().to_dict()
    geo_hour_mean = train_48.groupby("geo_hour")["demand"].mean().to_dict()
    hour_mean = train_48.groupby("hour")["demand"].mean().to_dict()
    weather_mean = train_48.groupby("Weather")["demand"].mean().to_dict()

    # Multi-Scale Hierarchical Aggregations
    geo_5_mean = train_48.groupby("geo_5")["demand"].mean().to_dict()
    geo_4_mean = train_48.groupby("geo_4")["demand"].mean().to_dict()
    road_slot_mean = train_48.groupby("road_slot")["demand"].mean().to_dict()
    weather_slot_mean = train_48.groupby("weather_slot")["demand"].mean().to_dict()

    for df in (train_df, predict_df):
        fallback = df["time_slot"].map(slot_mean).fillna(global_mean)
        df["geo_mean_d48"] = df["geohash_raw"].map(geo_mean).fillna(fallback)
        df["geo_std_d48"] = df["geohash_raw"].map(geo_std).fillna(0.0)
        df["geo_hour_mean_d48"] = df["geo_hour"].map(geo_hour_mean).fillna(df["geo_mean_d48"])
        df["hour_mean_d48"] = df["hour"].map(hour_mean).fillna(fallback)
        df["weather_mean_d48"] = df["Weather"].map(weather_mean).fillna(fallback)
        df["slot_mean_d48"] = fallback
        
        df["geo_5_mean_d48"] = df["geo_5"].map(geo_5_mean).fillna(fallback)
        df["geo_4_mean_d48"] = df["geo_4"].map(geo_4_mean).fillna(fallback)
        df["road_slot_mean_d48"] = df["road_slot"].map(road_slot_mean).fillna(fallback)
        df["weather_slot_mean_d48"] = df["weather_slot"].map(weather_slot_mean).fillna(fallback)

    freq_base = train_df.copy()
    for col in ["geohash_raw", "geo_hour", "road_hour"]:
        counts = freq_base[col].value_counts().to_dict()
        train_df[f"{col}_freq"] = train_df[col].map(counts).fillna(0.0)
        predict_df[f"{col}_freq"] = predict_df[col].map(counts).fillna(0.0)

    return train_df, predict_df, slot_mean


def add_day48_anchors(train_df, predict_df, slot_mean):
    train_48 = train_df[train_df["day"] == 48].copy()
    d48_demand_dict = train_48.set_index(["geohash_raw", "time_slot"])["demand"].to_dict()
    global_mean = train_48["demand"].mean()
    
    for df in (train_df, predict_df):
        fallback = df["time_slot"].map(slot_mean).fillna(global_mean)
        df["demand_prev_day"] = df.apply(lambda r: d48_demand_dict.get((r["geohash_raw"], r["time_slot"]), slot_mean.get(r["time_slot"], global_mean)) if r["day"] == 49 else np.nan, axis=1)
        
        for lag in [1, 2, 3, 4, 8, 12, 24]:
            df[f"lag_{lag}_d48"] = df.apply(lambda r: d48_demand_dict.get((r["geohash_raw"], r["time_slot"] - lag), slot_mean.get(max(0, r["time_slot"] - lag), global_mean)) if r["day"] == 49 else np.nan, axis=1)
            
        df["rolling_mean_4_d48"] = (df["demand_prev_day"] + df["lag_1_d48"] + df["lag_2_d48"] + df["lag_3_d48"]) / 4.0
        
        df["rolling_mean_8_d48"] = df["rolling_mean_4_d48"] * 0.5
        for lag in [4, 5, 6, 7]:
            df[f"lag_{lag}_d48_tmp"] = df.apply(lambda r: d48_demand_dict.get((r["geohash_raw"], r["time_slot"] - lag), slot_mean.get(max(0, r["time_slot"] - lag), global_mean)) if r["day"] == 49 else np.nan, axis=1)
            df["rolling_mean_8_d48"] += df[f"lag_{lag}_d48_tmp"] / 8.0
            
        df["rolling_mean_12_d48"] = df["rolling_mean_8_d48"] * (8.0/12.0)
        for lag in [8, 9, 10, 11]:
            df[f"lag_{lag}_d48_tmp"] = df.apply(lambda r: d48_demand_dict.get((r["geohash_raw"], r["time_slot"] - lag), slot_mean.get(max(0, r["time_slot"] - lag), global_mean)) if r["day"] == 49 else np.nan, axis=1)
            df["rolling_mean_12_d48"] += df[f"lag_{lag}_d48_tmp"] / 12.0
            
        lag_cols_4 = ["demand_prev_day", "lag_1_d48", "lag_2_d48", "lag_3_d48"]
        df["rolling_std_4_d48"] = df[lag_cols_4].std(axis=1)
        df["rolling_max_8_d48"] = df[["demand_prev_day", "lag_1_d48", "lag_2_d48", "lag_3_d48", "lag_4_d48", "lag_8_d48"]].max(axis=1)
        df["rolling_min_8_d48"] = df[["demand_prev_day", "lag_1_d48", "lag_2_d48", "lag_3_d48", "lag_4_d48", "lag_8_d48"]].min(axis=1)
        
        df["demand_prev_day"] = df["demand_prev_day"].fillna(fallback)
        for lag in [1, 2, 3, 4, 8, 12, 24]:
            df[f"lag_{lag}_d48"] = df[f"lag_{lag}_d48"].fillna(fallback)
        
        df["rolling_mean_4_d48"] = df["rolling_mean_4_d48"].fillna(fallback)
        df["rolling_mean_8_d48"] = df["rolling_mean_8_d48"].fillna(fallback)
        df["rolling_mean_12_d48"] = df["rolling_mean_12_d48"].fillna(fallback)
        df["rolling_std_4_d48"] = df["rolling_std_4_d48"].fillna(0.0)
        df["rolling_max_8_d48"] = df["rolling_max_8_d48"].fillna(fallback)
        df["rolling_min_8_d48"] = df["rolling_min_8_d48"].fillna(fallback)
        
        df.drop(columns=[c for c in df.columns if c.endswith("_tmp")], inplace=True, errors="ignore")
        
    return train_df, predict_df


def compute_same_day_features(df, history_maps, d48_history, slot_mean, alpha=1.0):
    day = int(df["day"].iloc[0])
    history = history_maps.get(day, {})
    fallback = df["time_slot"].map(slot_mean)
    
    # Checklist explicit lags: 1, 2, 4, 8, 12, 24, 48, 96
    lags = [1, 2, 4, 8, 12, 24, 48, 96]
    
    for lag in lags:
        l_preds = []
        for i, (geohash, slot) in enumerate(zip(df["geohash_raw"], df["time_slot"])):
            val = history.get((geohash, slot - lag), np.nan)
            if not pd.isna(val) and val is not None:
                if day == 49 and slot - lag >= 9: # predictions start at slot 9
                    d48_slot = slot - lag
                    if d48_slot < 0:
                        d48_slot = d48_slot + 96
                    d48_val = d48_history.get((geohash, d48_slot), fallback.iloc[i])
                    val = alpha * val + (1.0 - alpha) * d48_val
            else:
                d48_slot = slot - lag
                if d48_slot < 0:
                    d48_slot = d48_slot + 96
                val = d48_history.get((geohash, d48_slot), fallback.iloc[i])
            l_preds.append(np.clip(val, 0.0, 1.0))
        df[f"same_day_lag_{lag}"] = l_preds
        
    # Checklist explicit rolling statistics: rolling mean, rolling std, rolling max
    df["rolling_mean_same"] = (df["same_day_lag_1"] + df["same_day_lag_2"] + df["same_day_lag_4"] + df["same_day_lag_8"]) / 4.0
    
    lag_cols_4 = [f"same_day_lag_{l}" for l in [1, 2, 4]]
    lag_cols_8 = [f"same_day_lag_{l}" for l in [1, 2, 4, 8]]
    df["rolling_std_same"] = df[lag_cols_4].std(axis=1).fillna(0.0)
    df["rolling_max_same"] = df[lag_cols_8].max(axis=1)
    
    return df


def encode_categories(train_df, predict_df):
    cat_cols = [
        "Weather", "RoadType", "LargeVehicles", "Landmarks",
        "geo_4", "geo_5", "geohash_encoded",
        "road_hour", "weather_hour", "lane_hour", "geo_hour", "geo_time_slot"
    ]
    
    le_geohash = LabelEncoder()
    combined_geo = pd.concat([train_df["geohash_raw"].astype(str), predict_df["geohash_raw"].astype(str)])
    le_geohash.fit(combined_geo)
    train_df["geohash_encoded"] = le_geohash.transform(train_df["geohash_raw"].astype(str))
    predict_df["geohash_encoded"] = le_geohash.transform(predict_df["geohash_raw"].astype(str))
    
    for col in cat_cols:
        if col == "geohash_encoded":
            continue
        le = LabelEncoder()
        combined = pd.concat([train_df[col].astype(str), predict_df[col].astype(str)])
        le.fit(combined)
        train_df[col] = le.transform(train_df[col].astype(str))
        predict_df[col] = le.transform(predict_df[col].astype(str))
        
    return train_df, predict_df

# ==============================================================================
# 2. Features Space & Model Definitions
# ==============================================================================

FEATURES = [
    "geo_4", "geo_5", "geohash_encoded", "time_slot_sin", "time_slot_cos",
    "lat", "lon", "RoadType", "NumberofLanes", "LargeVehicles", "Landmarks", "Temperature", "Weather",
    # Target encoding raw columns (dropped prior to fitting models)
    "geo_hour", "road_hour", "geo_time_slot",
    # Same-day recursive lags matching checklist exactly
    "same_day_lag_1", "same_day_lag_2", "same_day_lag_4", "same_day_lag_8", "same_day_lag_12", "same_day_lag_24", "same_day_lag_48", "same_day_lag_96",
    # Checklist rolling statistics
    "rolling_mean_same", "rolling_std_same", "rolling_max_same",
    # Day 48 statistics
    "geo_mean_d48", "geo_std_d48", "geo_hour_mean_d48", "hour_mean_d48", "weather_mean_d48", "slot_mean_d48",
    "geo_5_mean_d48", "geo_4_mean_d48", "road_slot_mean_d48", "weather_slot_mean_d48",
    # Interaction key frequencies
    "geohash_raw_freq", "geo_hour_freq", "road_hour_freq",
    # Temporal engineered states
    "is_weekend", "is_peak_hour", "is_night", "is_office_hour", "part_of_day", "rush_hour", "spatial_cluster"
]

TE_COLS = ["geohash_encoded", "geo_hour", "road_hour", "geo_time_slot"]


def run_training(train_df):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_48 = train_df[train_df["day"] == 48].copy()
    train_49 = train_df[train_df["day"] == 49].copy()
    
    models_by_fold = []
    fold_te_dicts = []
    
    print("\nTraining 5-Fold Ensemble Models strictly with safe OOF Target Encoding...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_49)):
        print(f"--- Training FOLD {fold+1} ---")
        train_49_fold = train_49.iloc[train_idx].copy()
        val_fold = train_49.iloc[val_idx].copy()
        
        combined_train = pd.concat([train_48, train_49_fold], ignore_index=True)
        
        X_tr = combined_train[FEATURES].copy()
        y_tr = combined_train["demand"].copy()
        X_va = val_fold[FEATURES].copy()
        y_va = val_fold["demand"].copy()
        
        # 100% Leakage-Free Out-of-fold target encoding for training
        te_dicts = {}
        for col in TE_COLS:
            X_tr[f"{col}_te"] = np.nan
            X_va[f"{col}_te"] = np.nan
            
            global_mean = y_tr.mean()
            
            # Sub-split combined_train to do OOF target encoding
            sub_kf = KFold(n_splits=5, shuffle=True, random_state=42)
            for sub_tr_idx, sub_val_idx in sub_kf.split(X_tr):
                sub_y_tr_fold = y_tr.iloc[sub_tr_idx]
                sub_X_tr_fold = X_tr.iloc[sub_tr_idx]
                
                # Compute map
                te_map = sub_y_tr_fold.groupby(sub_X_tr_fold[col]).mean().to_dict()
                
                # Map to validation fold
                X_tr.iloc[sub_val_idx, X_tr.columns.get_loc(f"{col}_te")] = (
                    X_tr.iloc[sub_val_idx][col].map(te_map).fillna(global_mean)
                )
                
            X_tr[f"{col}_te"] = X_tr[f"{col}_te"].fillna(global_mean)
            
            # Map X_va using the full training target map
            full_te_map = y_tr.groupby(X_tr[col]).mean().to_dict()
            X_va[f"{col}_te"] = X_va[col].map(full_te_map).fillna(global_mean)
            
            te_dicts[col] = (full_te_map, global_mean)
            
        fold_te_dicts.append(te_dicts)
        
        X_tr_fit = X_tr.drop(columns=TE_COLS)
        X_va_fit = X_va.drop(columns=TE_COLS)
        
        # Fine-tuned final estimators
        lgb = LGBMRegressor(
            n_estimators=1500,
            learning_rate=0.025,
            num_leaves=63,
            random_state=42,
            verbose=-1,
        )
        lgb.fit(X_tr_fit, y_tr, eval_set=[(X_va_fit, y_va)], callbacks=[early_stopping(50, verbose=False)])
        
        xgb = XGBRegressor(
            n_estimators=1500,
            learning_rate=0.025,
            max_depth=6,
            early_stopping_rounds=50,
            random_state=42,
            verbosity=0,
            n_jobs=1,
        )
        xgb.fit(X_tr_fit, y_tr, eval_set=[(X_va_fit, y_va)], verbose=False)
        
        cat = CatBoostRegressor(
            iterations=1600,
            learning_rate=0.025,
            depth=6,
            early_stopping_rounds=50,
            random_state=42,
            allow_writing_files=False,
            verbose=0,
        )
        cat.fit(X_tr_fit, y_tr, eval_set=[(X_va_fit, y_va)], verbose=False)
        
        models_by_fold.append((lgb, xgb, cat))
        
    return models_by_fold, fold_te_dicts


def blend_predictions(models, X_fit):
    lgb, xgb, cat = models
    p_lgb = lgb.predict(X_fit)
    p_xgb = xgb.predict(X_fit)
    p_cat = cat.predict(X_fit)
    
    # Blending weights tuned for robustness (15% LGBM, 45% XGBoost, 40% CatBoost)
    return np.clip(0.15 * p_lgb + 0.45 * p_xgb + 0.40 * p_cat, 0.0, 1.0)

# ==============================================================================
# 3. Main Pipeline: Final Inference & Submission Generation
# ==============================================================================

def make_submission():
    print("\n=== STARTING ULTIMATE FORECASTING SUBMISSION PIPELINE (V5) ===")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    train_df, test_df = clean_columns(train_df, test_df)
    train_df, test_df, slot_mean = add_static_stats(train_df, test_df)
    train_df, test_df = add_day48_anchors(train_df, test_df, slot_mean)
    
    d48_history = train_df[train_df["day"] == 48].set_index(
        ["geohash_raw", "time_slot"]
    )["demand"].to_dict()
    
    history_maps = {
        48: d48_history,
        49: train_df[train_df["day"] == 49].set_index(["geohash_raw", "time_slot"])["demand"].to_dict(),
    }
    
    # Add same-day recursive features for training data (alpha=1.0)
    print("Computing same-day recursive lag features...")
    train_features = pd.concat(
        [
            compute_same_day_features(part.copy(), history_maps, d48_history, slot_mean, alpha=1.0)
            for _, part in train_df.groupby("day", sort=True)
        ],
        ignore_index=True,
    )
    
    train_features, test_df = encode_categories(train_features, test_df)
    
    # Train the fold models
    models_by_fold, fold_te_dicts = run_training(train_features)
    
    # 4. Perform sequential recursive forecasting on test set slot-by-slot
    print("\nGenerating final recursive predictions slot-by-slot on test set...")
    d49_history = train_df[train_df["day"] == 49].set_index(
        ["geohash_raw", "time_slot"]
    )["demand"].to_dict()

    ordered_test = test_df.sort_values(["day", "time_slot", "Index"])
    output = []

    # Sequence slot-by-slot starting at slot 9 (2:15 AM) recursively!
    for _, slot_frame in ordered_test.groupby(["day", "time_slot"], sort=True):
        batch = slot_frame.copy()
        slot = int(batch["time_slot"].iloc[0])
        fallback = slot_mean.get(slot, train_features["demand"].mean())
        
        # Build recursive features using alpha=1.0
        batch = compute_same_day_features(batch, {49: d49_history}, d48_history, slot_mean, alpha=1.0)
        
        # Predict using 5-fold ensemble averaging with fold-safe mappings!
        fold_preds = []
        for fold in range(5):
            te_dicts = fold_te_dicts[fold]
            X_batch = batch[FEATURES].copy()
            
            for col in TE_COLS:
                target_mean, global_mean = te_dicts[col]
                X_batch[f"{col}_te"] = X_batch[col].map(target_mean).fillna(global_mean)
                
            X_batch_fit = X_batch.drop(columns=TE_COLS)
            pred = blend_predictions(models_by_fold[fold], X_batch_fit)
            fold_preds.append(pred)
            
        predictions = np.mean(fold_preds, axis=0)
        
        for index_value, geohash, prediction in zip(
            batch["Index"], batch["geohash_raw"].astype(str), predictions
        ):
            d49_history[(geohash, slot)] = float(prediction)
            output.append((int(index_value), float(prediction)))

    submission = pd.DataFrame(output, columns=["Index", "demand"]).sort_values("Index")
    try:
        submission.to_csv(SUBMISSION_PATH, index=False)
    except PermissionError:
        alternative_path = "submission_anchored.csv"
        submission.to_csv(alternative_path, index=False)
        print(f"\n[WARNING] Permission denied to write to '{SUBMISSION_PATH}'.")
        print(f"[SUCCESS] Saved submission instead to: '{alternative_path}'")
    return submission


if __name__ == "__main__":
    submission = make_submission()
    print(f"\n[SUCCESS] Successfully generated final bagged recursive target-encoded predictions!")
    print(f"Saved {SUBMISSION_PATH} with {len(submission)} predictions.")
    print("=======================================================================")
