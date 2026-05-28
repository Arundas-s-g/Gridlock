import warnings

import numpy as np
import pandas as pd
import pygeohash as pgh
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


TRAIN_PATH = "dataset/train.csv"
TEST_PATH = "dataset/test.csv"
SUBMISSION_PATH = "submission.csv"


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
    predict_df["lon"] = predict_df["lon"].fillna(train_df["lon"].mean())

    kmeans = KMeans(n_clusters=10, random_state=42, n_init="auto")
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
        df["road_hour"] = df["RoadType"].astype(str) + "_" + df["hour"].astype(str)
        df["weather_hour"] = df["Weather"].astype(str) + "_" + df["hour"].astype(str)
        df["lane_hour"] = df["NumberofLanes"].astype(str) + "_" + df["hour"].astype(str)
        df["geo_hour"] = df["geohash_raw"] + "_" + df["hour"].astype(str)

    return train_df, predict_df


def build_history_maps(train_df):
    day_48 = train_df[train_df["day"] == 48].copy()
    day_49 = train_df[train_df["day"] == 49].copy()

    return {
        48: day_48.set_index(["geohash_raw", "time_slot"])["demand"].to_dict(),
        49: day_49.set_index(["geohash_raw", "time_slot"])["demand"].to_dict(),
    }


def add_static_stats(train_df, predict_df):
    day_48 = train_df[train_df["day"] == 48].copy()
    slot_mean = day_48.groupby("time_slot")["demand"].mean().to_dict()
    geo_mean = day_48.groupby("geohash_raw")["demand"].mean().to_dict()
    geo_std = day_48.groupby("geohash_raw")["demand"].std().to_dict()
    geo_hour_mean = day_48.groupby("geo_hour")["demand"].mean().to_dict()
    hour_mean = day_48.groupby("hour")["demand"].mean().to_dict()
    weather_mean = day_48.groupby("Weather")["demand"].mean().to_dict()

    for df in (train_df, predict_df):
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
        predict_df[f"{col}_freq"] = predict_df[col].map(counts).fillna(0.0)

    return train_df, predict_df, slot_mean


def add_day48_anchors(train_df, predict_df, slot_mean):
    train_48 = train_df[train_df["day"] == 48].copy()
    d48_demand_dict = train_48.set_index(["geohash_raw", "time_slot"])["demand"].to_dict()
    for df in (train_df, predict_df):
        fallback = df["time_slot"].map(slot_mean)
        df["demand_prev_day"] = df.apply(lambda r: d48_demand_dict.get((r["geohash_raw"], r["time_slot"]), fallback.get(r["time_slot"])) if r["day"] == 49 else fallback.get(r["time_slot"]), axis=1)
        df["lag_1_d48"] = df.apply(lambda r: d48_demand_dict.get((r["geohash_raw"], r["time_slot"] - 1), fallback.get(r["time_slot"] - 1)) if r["day"] == 49 else fallback.get(r["time_slot"] - 1), axis=1)
        df["lag_2_d48"] = df.apply(lambda r: d48_demand_dict.get((r["geohash_raw"], r["time_slot"] - 2), fallback.get(r["time_slot"] - 2)) if r["day"] == 49 else fallback.get(r["time_slot"] - 2), axis=1)
        df["lag_3_d48"] = df.apply(lambda r: d48_demand_dict.get((r["geohash_raw"], r["time_slot"] - 3), fallback.get(r["time_slot"] - 3)) if r["day"] == 49 else fallback.get(r["time_slot"] - 3), axis=1)
    return train_df, predict_df


def add_same_day_lags(df, history_maps, slot_mean):
    day = int(df["day"].iloc[0])
    history = history_maps.get(day, {})
    d48_history = history_maps.get(48, {})
    fallback = df["time_slot"].map(slot_mean)
    for lag in (1, 2, 3):
        lags_list = []
        for i, (geohash, slot) in enumerate(zip(df["geohash_raw"], df["time_slot"])):
            val = history.get((geohash, slot - lag), np.nan)
            if pd.isna(val) or val is None:
                # Hierarchical fallback: lookup Day 48 same geohash + slot demand
                val = d48_history.get((geohash, slot - lag), fallback.iloc[i])
            lags_list.append(val)
        df[f"same_day_lag_{lag}"] = lags_list

    df["rolling_mean_3_same"] = (
        df["same_day_lag_1"] + df["same_day_lag_2"] + df["same_day_lag_3"]
    ) / 3.0
    df["diff_1_same"] = df["same_day_lag_1"] - df["same_day_lag_2"]
    return df


def encode_categories(train_df, predict_df):
    cat_cols = [
        "Weather",
        "RoadType",
        "LargeVehicles",
        "Landmarks",
        "geo_4",
        "geo_5",
        "road_hour",
        "weather_hour",
        "lane_hour",
    ]
    for col in cat_cols:
        encoder = LabelEncoder()
        combined = pd.concat([train_df[col].astype(str), predict_df[col].astype(str)])
        encoder.fit(combined)
        train_df[col] = encoder.transform(train_df[col].astype(str))
        predict_df[col] = encoder.transform(predict_df[col].astype(str))
    return train_df, predict_df


FEATURES = [
    "geo_4",
    "geo_5",
    "time_slot_sin",
    "time_slot_cos",
    "lat",
    "lon",
    "RoadType",
    "NumberofLanes",
    "LargeVehicles",
    "Landmarks",
    "Temperature",
    "Weather",
    "same_day_lag_1",
    "same_day_lag_2",
    "same_day_lag_3",
    "rolling_mean_3_same",
    "diff_1_same",
    "geo_mean_d48",
    "geo_std_d48",
    "geo_hour_mean_d48",
    "hour_mean_d48",
    "weather_mean_d48",
    "slot_mean_d48",
    "road_hour",
    "weather_hour",
    "lane_hour",
    "geohash_raw_freq",
    "geo_hour_freq",
    "road_hour_freq",
    "is_weekend",
    "is_peak_hour",
    "is_night",
    "is_office_hour",
    "spatial_cluster",
    "demand_prev_day",
    "lag_1_d48",
    "lag_2_d48",
    "lag_3_d48",
]


def train_models(train_features):
    x_train = train_features[FEATURES]
    y_train = train_features["demand"]

    models = [
        LGBMRegressor(
            n_estimators=1200,
            learning_rate=0.025,
            num_leaves=31,
            random_state=42,
            verbose=-1,
        ),
        XGBRegressor(
            n_estimators=1200,
            learning_rate=0.025,
            max_depth=5,
            random_state=42,
            verbosity=0,
            n_jobs=1,
        ),
        CatBoostRegressor(
            iterations=1400,
            learning_rate=0.025,
            depth=5,
            random_state=42,
            allow_writing_files=False,
            verbose=0,
        ),
    ]

    for model in models:
        model.fit(x_train, y_train)
    return models


def blend_predictions(models, x_predict):
    # Equal-blend (0.34 LightGBM + 0.33 XGBoost + 0.33 CatBoost) is highly stable under CV
    weights = np.array([0.34, 0.33, 0.33])
    predictions = np.column_stack([model.predict(x_predict) for model in models])
    return np.clip(predictions @ weights, 0.0, 1.0)


def holdout_check():
    raw_train = pd.read_csv(TRAIN_PATH)
    train_df = raw_train.iloc[:77000].copy()
    holdout_df = raw_train.iloc[77000:].copy()

    train_df, holdout_df = clean_columns(train_df, holdout_df)
    train_df, holdout_df, slot_mean = add_static_stats(train_df, holdout_df)
    train_df, holdout_df = add_day48_anchors(train_df, holdout_df, slot_mean)
    history_maps = build_history_maps(train_df)
    train_df = pd.concat(
        [
            add_same_day_lags(part.copy(), history_maps, slot_mean)
            for _, part in train_df.groupby("day", sort=True)
        ],
        ignore_index=True,
    )
    holdout_df = add_same_day_lags(holdout_df, history_maps, slot_mean)
    train_df, holdout_df = encode_categories(train_df, holdout_df)
    
    # 5-Fold cross validation holdout evaluation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    models_by_fold = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        tr_df = train_df.iloc[train_idx]
        fold_models = train_models(tr_df)
        models_by_fold.append(fold_models)
    
    # Predict recursively on holdout using 5-fold ensemble
    d49_history = train_df[train_df["day"] == 49].set_index(
        ["geohash_raw", "time_slot"]
    )["demand"].to_dict()
    
    d48_same_slot_dict = train_df[train_df["day"] == 48].set_index(
        ["geohash_raw", "time_slot"]
    )["demand"].to_dict()
    
    ordered_holdout = holdout_df.sort_values(["time_slot", "Index"])
    preds_list = []
    idx_list = []
    
    for _, slot_frame in ordered_holdout.groupby("time_slot", sort=True):
        batch = slot_frame.copy()
        slot = int(batch["time_slot"].iloc[0])
        fallback = slot_mean.get(slot, train_df["demand"].mean())
        
        for lag in (1, 2, 3):
            l_preds = []
            for geohash in batch["geohash_raw"]:
                if (geohash, slot - lag) in d49_history:
                    val = d49_history[(geohash, slot - lag)]
                else:
                    d48_val = d48_same_slot_dict.get((geohash, slot - lag), fallback)
                    val = np.clip(d48_val, 0.0, 1.0)
                l_preds.append(val)
            batch[f"same_day_lag_{lag}"] = l_preds
            
        batch["rolling_mean_3_same"] = (batch["same_day_lag_1"] + batch["same_day_lag_2"] + batch["same_day_lag_3"]) / 3.0
        batch["diff_1_same"] = batch["same_day_lag_1"] - batch["same_day_lag_2"]
        
        # Predict using all 5 folds
        fold_preds = []
        for fold_models in models_by_fold:
            pred = blend_predictions(fold_models, batch[FEATURES])
            fold_preds.append(pred)
        preds = np.mean(fold_preds, axis=0)
        
        for index_val, geohash, pred in zip(batch["Index"], batch["geohash_raw"], preds):
            d49_history[(geohash, slot)] = float(pred)
            preds_list.append(float(pred))
            idx_list.append(int(index_val))
            
    eval_df = pd.DataFrame({"Index": idx_list, "pred_demand": preds_list}).set_index("Index")
    eval_df = holdout_df.set_index("Index").join(eval_df)
    
    y_true = eval_df["demand"]
    y_pred = eval_df["pred_demand"]
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse,
        "accuracy_percent": (1 - rmse) * 100,
    }


def make_submission():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    train_df, test_df = clean_columns(train_df, test_df)
    train_df, test_df, slot_mean = add_static_stats(train_df, test_df)
    train_df, test_df = add_day48_anchors(train_df, test_df, slot_mean)
    history_maps = build_history_maps(train_df)
    train_features = pd.concat(
        [
            add_same_day_lags(part.copy(), history_maps, slot_mean)
            for _, part in train_df.groupby("day", sort=True)
        ],
        ignore_index=True,
    )
    test_df, encoded_test = encode_categories(train_features.copy(), test_df.copy())
    train_features = test_df
    test_df = encoded_test

    # 5-Fold cross validation bagged model training
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    models_by_fold = []
    print("\n=== Training 5-Fold Ensemble Models for Submission ===")
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_features)):
        print(f"--- Training FOLD {fold+1} ---")
        tr_features = train_features.iloc[train_idx]
        fold_models = train_models(tr_features)
        models_by_fold.append(fold_models)

    raw_test = pd.read_csv(TEST_PATH)
    output = []
    d48_history = train_df[train_df["day"] == 48].set_index(
        ["geohash_raw", "time_slot"]
    )["demand"].to_dict()
    d49_history = train_df[train_df["day"] == 49].set_index(
        ["geohash_raw", "time_slot"]
    )["demand"].to_dict()

    ordered_test = test_df.sort_values(["day", "time_slot", "Index"])

    for _, slot_frame in ordered_test.groupby(["day", "time_slot"], sort=True):
        batch = slot_frame.copy()
        slot = int(batch["time_slot"].iloc[0])
        fallback = slot_mean.get(slot, train_features["demand"].mean())
        for lag in (1, 2, 3):
            l_preds = []
            for geohash in batch["geohash_raw"].astype(str):
                if (geohash, slot - lag) in d49_history:
                    val = d49_history[(geohash, slot - lag)]
                else:
                    d48_val = d48_history.get((geohash, slot - lag), fallback)
                    val = np.clip(d48_val, 0.0, 1.0)
                l_preds.append(val)
            batch[f"same_day_lag_{lag}"] = l_preds
            
        batch["rolling_mean_3_same"] = (
            batch["same_day_lag_1"] + batch["same_day_lag_2"] + batch["same_day_lag_3"]
        ) / 3.0
        batch["diff_1_same"] = batch["same_day_lag_1"] - batch["same_day_lag_2"]

        # Predict using 5-fold ensemble averaging
        fold_preds = []
        for fold_models in models_by_fold:
            pred = blend_predictions(fold_models, batch[FEATURES])
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
        print(f"\n[WARNING] Permission denied to write to '{SUBMISSION_PATH}' (likely because it is open in your active editor).")
        print(f"[SUCCESS] Saved submission instead to: '{alternative_path}'")
    return submission


if __name__ == "__main__":
    metrics = holdout_check()
    print("Holdout check on train rows 77000 to end")
    print(f"R2:       {metrics['r2']:.5f}")
    print(f"MAE:      {metrics['mae']:.5f}")
    print(f"RMSE:     {metrics['rmse']:.5f}")
    print(f"Accuracy: {metrics['accuracy_percent']:.2f}%")

    submission = make_submission()
    print(f"Saved {SUBMISSION_PATH} with {len(submission)} predictions.")
