import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import final_submission_v3 as pipeline


RANDOM_STATE = 42
HOLDOUT_SIZE = 500


def predict_holdout(train_raw, holdout_raw):
    train_df, holdout_df = pipeline.clean_columns(train_raw.copy(), holdout_raw.copy())
    train_df, holdout_df, slot_mean = pipeline.add_static_stats(train_df, holdout_df)
    train_df, holdout_df = pipeline.add_day48_anchors(train_df, holdout_df, slot_mean)
    history_maps = pipeline.build_history_maps(train_df)

    train_features = pd.concat(
        [
            pipeline.add_same_day_lags(part.copy(), history_maps, slot_mean)
            for _, part in train_df.groupby("day", sort=True)
        ],
        ignore_index=True,
    )
    holdout_features = pd.concat(
        [
            pipeline.add_same_day_lags(part.copy(), history_maps, slot_mean)
            for _, part in holdout_df.groupby("day", sort=True)
        ],
        ignore_index=True,
    )

    train_features, holdout_features = pipeline.encode_categories(
        train_features, holdout_features
    )
    models = pipeline.train_models(train_features)
    predictions = pipeline.blend_predictions(models, holdout_features[pipeline.FEATURES])
    return holdout_features["demand"].to_numpy(), predictions


def report(name, actual, predicted):
    r2 = r2_score(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    print(name)
    print(f"R2:    {r2:.5f}")
    print(f"Score: {max(0, 100 * r2):.2f}")
    print(f"MAE:   {mae:.5f}")
    print(f"RMSE:  {rmse:.5f}")
    print()


def predict_future_recursively(train_raw, future_raw, use_anchor=False):
    train_df, future_df = pipeline.clean_columns(train_raw.copy(), future_raw.copy())
    train_df, future_df, slot_mean = pipeline.add_static_stats(train_df, future_df)
    train_df, future_df = pipeline.add_day48_anchors(train_df, future_df, slot_mean)
    history_maps = pipeline.build_history_maps(train_df)

    train_features = pd.concat(
        [
            pipeline.add_same_day_lags(part.copy(), history_maps, slot_mean)
            for _, part in train_df.groupby("day", sort=True)
        ],
        ignore_index=True,
    )
    train_features, future_features = pipeline.encode_categories(
        train_features.copy(), future_df.copy()
    )
    models = pipeline.train_models(train_features)

    d49_history = train_df[train_df["day"] == 49].set_index(
        ["geohash_raw", "time_slot"]
    )["demand"].to_dict()

    output = []
    ordered = future_features.sort_values(["day", "time_slot", "Index"])
    for _, slot_frame in ordered.groupby(["day", "time_slot"], sort=True):
        batch = slot_frame.copy()
        slot = int(batch["time_slot"].iloc[0])
        fallback = slot_mean.get(slot, train_features["demand"].mean())
        for lag in (1, 2, 3):
            batch[f"same_day_lag_{lag}"] = [
                d49_history.get((geohash, slot - lag), fallback)
                for geohash in batch["geohash_raw"].astype(str)
            ]
        batch["rolling_mean_3_same"] = (
            batch["same_day_lag_1"] + batch["same_day_lag_2"] + batch["same_day_lag_3"]
        ) / 3.0
        batch["diff_1_same"] = batch["same_day_lag_1"] - batch["same_day_lag_2"]

        predictions = pipeline.blend_predictions(models, batch[pipeline.FEATURES])
        for index_value, geohash, prediction in zip(
            batch["Index"], batch["geohash_raw"].astype(str), predictions
        ):
            d49_history[(geohash, slot)] = float(prediction)
            output.append((int(index_value), float(prediction)))

    submission = pd.DataFrame(output, columns=["Index", "demand"]).sort_values("Index")
    if use_anchor:
        submission = pipeline.apply_day48_anchor(train_df, future_raw.copy(), submission)

    actual = future_raw.set_index("Index").loc[submission["Index"], "demand"].to_numpy()
    return actual, submission["demand"].to_numpy()


def main():
    raw = pd.read_csv(pipeline.TRAIN_PATH)

    holdout_index = raw.sample(HOLDOUT_SIZE, random_state=RANDOM_STATE).index
    train_raw = raw.drop(index=holdout_index).copy()
    holdout_raw = raw.loc[holdout_index].copy()
    actual, predicted = predict_holdout(train_raw, holdout_raw)
    report("Random 500-row holdout, excluded from training", actual, predicted)

    dt = pd.to_datetime(raw["timestamp"], format="%H:%M")
    raw = raw.assign(time_slot=dt.dt.hour * 4 + dt.dt.minute // 15)

    train_like = raw[
        (raw["day"] == 48) | ((raw["day"] == 49) & (raw["time_slot"] <= 5))
    ].drop(columns=["time_slot"])
    future_like = raw[
        (raw["day"] == 49) & (raw["time_slot"].between(6, 8))
    ].drop(columns=["time_slot"])
    actual, predicted = predict_holdout(train_like, future_like)
    report("Future-time proxy: train through day49 slot 5, validate slots 6-8", actual, predicted)

    actual, predicted = predict_future_recursively(train_like, future_like, use_anchor=False)
    report("Future-time proxy with recursive prediction feed", actual, predicted)

    actual, predicted = predict_future_recursively(train_like, future_like, use_anchor=True)
    report("Future-time proxy with recursive feed + day48 anchor", actual, predicted)


if __name__ == "__main__":
    main()
