import pandas as pd
import numpy as np
import pygeohash as pgh
from lightgbm import LGBMRegressor, early_stopping
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')

print("=== Starting Final Submission Pipeline ===")

# 1. Load data
print("Loading train.csv and test.csv...")
train = pd.read_csv("dataset/train.csv")
test = pd.read_csv("dataset/test.csv")

# 2. Convert timestamp to time slot (0-95)
def process_time(df):
    df['dt'] = pd.to_datetime(df['timestamp'], format='%H:%M')
    df['hour'] = df['dt'].dt.hour
    df['minute'] = df['dt'].dt.minute
    df['time_slot'] = df['hour'] * 4 + df['minute'] // 15
    df['time_slot_sin'] = np.sin(2 * np.pi * df['time_slot'] / 96)
    df['time_slot_cos'] = np.cos(2 * np.pi * df['time_slot'] / 96)
    df.drop(columns=['dt'], inplace=True)
    return df

train = process_time(train)
test = process_time(test)

# 3. Decode geohash to lat/lon
print("Decoding geohashes to lat/lon...")
geohashes = pd.concat([train['geohash'], test['geohash']]).unique()
geo_coords = {}
for g in geohashes:
    try:
        lat, lon = pgh.decode(g)
        geo_coords[g] = (lat, lon)
    except Exception:
        geo_coords[g] = (np.nan, np.nan)

train['lat'] = train['geohash'].map(lambda g: geo_coords[g][0])
train['lon'] = train['geohash'].map(lambda g: geo_coords[g][1])
test['lat'] = test['geohash'].map(lambda g: geo_coords[g][0])
test['lon'] = test['geohash'].map(lambda g: geo_coords[g][1])

# 4. Fill missing values
print("Imputing missing values...")
temp_median = train['Temperature'].median()
geo_temp_median = train.groupby('geohash')['Temperature'].median().to_dict()

def impute_temp(row):
    if pd.isna(row['Temperature']):
        return geo_temp_median.get(row['geohash'], temp_median)
    return row['Temperature']

train['Temperature'] = train.apply(impute_temp, axis=1)
test['Temperature'] = test.apply(impute_temp, axis=1)

train['Weather'] = train['Weather'].fillna("Unknown")
test['Weather'] = test['Weather'].fillna("Unknown")
train['RoadType'] = train['RoadType'].fillna("Unknown")
test['RoadType'] = test['RoadType'].fillna("Unknown")

# 5. Geohash prefixes
train['geo_4'] = train['geohash'].astype(str).str[:4]
train['geo_5'] = train['geohash'].astype(str).str[:5]
test['geo_4'] = test['geohash'].astype(str).str[:4]
test['geo_5'] = test['geohash'].astype(str).str[:5]

# 6. Categorical encoding
print("Encoding categorical columns...")
cat_cols = ['Weather', 'RoadType', 'LargeVehicles', 'Landmarks', 'geohash', 'geo_4', 'geo_5']
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col].astype(str), test[col].astype(str)])
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# 7. Construct demand_prev_day feature
print("Constructing Day 48 demand lookup features...")
train_48 = train[train['day'] == 48].copy()
train_49 = train[train['day'] == 49].copy()

d48_demand_dict = train_48.set_index(['geohash', 'time_slot'])['demand'].to_dict()

train_49['demand_prev_day'] = train_49.apply(lambda r: d48_demand_dict.get((r['geohash'], r['time_slot']), np.nan), axis=1)
test['demand_prev_day'] = test.apply(lambda r: d48_demand_dict.get((r['geohash'], r['time_slot']), np.nan), axis=1)

# Group by statistics on Day 48
geo_mean_d48 = train_48.groupby('geohash')['demand'].mean().to_dict()
geo_std_d48 = train_48.groupby('geohash')['demand'].std().to_dict()
slot_mean_d48 = train_48.groupby('time_slot')['demand'].mean().to_dict()

for df in [train_49, test]:
    df['geo_mean_d48'] = df['geohash'].map(geo_mean_d48)
    df['geo_std_d48'] = df['geohash'].map(geo_std_d48)
    df['slot_mean_d48'] = df['time_slot'].map(slot_mean_d48)
    
    # Impute missing values for the lags using slot mean
    df['demand_prev_day'] = df['demand_prev_day'].fillna(df['slot_mean_d48'])

# 8. Define features
features = [
    'geohash', 'geo_4', 'geo_5', 'time_slot_sin', 'time_slot_cos',
    'lat', 'lon', 'RoadType', 'NumberofLanes', 'LargeVehicles', 'Landmarks',
    'Temperature', 'Weather', 'demand_prev_day',
    'geo_mean_d48', 'geo_std_d48', 'slot_mean_d48'
]

X = train_49[features]
y = train_49['demand']
X_test = test[features]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Out-of-fold and test predictions arrays
oof_preds = np.zeros(len(train_49))
test_preds = np.zeros(len(test))

lgb_oof = np.zeros(len(train_49))
xgb_oof = np.zeros(len(train_49))
cat_oof = np.zeros(len(train_49))

print("\n=== Training models with 5-Fold Cross Validation ===")
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n--- FOLD {fold+1} ---")
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    # Train LightGBM
    lgb = LGBMRegressor(n_estimators=3000, learning_rate=0.03, random_state=42, verbose=-1)
    lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[early_stopping(stopping_rounds=100, verbose=False)]
    )
    pred_lgb = lgb.predict(X_va)
    lgb_oof[val_idx] = pred_lgb
    
    # Train XGBoost
    xgb = XGBRegressor(n_estimators=3000, learning_rate=0.03, early_stopping_rounds=100, random_state=42, verbosity=0)
    xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )
    pred_xgb = xgb.predict(X_va)
    xgb_oof[val_idx] = pred_xgb
    
    # Train CatBoost
    cat = CatBoostRegressor(iterations=3000, learning_rate=0.03, early_stopping_rounds=100, random_state=42, verbose=0)
    cat.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )
    pred_cat = cat.predict(X_va)
    cat_oof[val_idx] = pred_cat
    
    # Ensemble for validation
    pred_ens = (pred_lgb + pred_xgb + pred_cat) / 3.0
    oof_preds[val_idx] = pred_ens
    
    # Predict on test
    test_preds += (lgb.predict(X_test) + xgb.predict(X_test) + cat.predict(X_test)) / 15.0
    
    print(f"Fold {fold+1} R2 scores:")
    print(f"  LightGBM R2: {r2_score(y_va, pred_lgb):.5f}")
    print(f"  XGBoost R2:  {r2_score(y_va, pred_xgb):.5f}")
    print(f"  CatBoost R2: {r2_score(y_va, pred_cat):.5f}")
    print(f"  Ensemble R2: {r2_score(y_va, pred_ens):.5f}")

print("\n=== Cross-Validation Summary ===")
print(f"Overall LightGBM OOF R2: {r2_score(y, lgb_oof):.5f}")
print(f"Overall XGBoost OOF R2:  {r2_score(y, xgb_oof):.5f}")
print(f"Overall CatBoost OOF R2: {r2_score(y, cat_oof):.5f}")
print(f"Overall Ensemble OOF R2: {r2_score(y, oof_preds):.5f}")

# Post-processing: clip test predictions to [0, 1]
test_preds_clipped = np.clip(test_preds, 0.0, 1.0)

# Create submission
submission = pd.DataFrame({
    'Index': test['Index'],
    'demand': test_preds_clipped
})

# Verify constraints
print("\n=== Verifying Submission Format ===")
print("Submission shape:", submission.shape)
print("Is null count > 0?", submission.isnull().sum().sum() > 0)
print("Min prediction value:", submission['demand'].min())
print("Max prediction value:", submission['demand'].max())
print("First 5 rows of submission:")
print(submission.head())

# Save to CSV
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"\nSaved submission to {submission_path}")
print("=== Final Pipeline Completed Successfully ===")
