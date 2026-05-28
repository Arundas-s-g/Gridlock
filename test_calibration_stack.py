import pandas as pd
import numpy as np
import pygeohash as pgh
from lightgbm import LGBMRegressor, early_stopping
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_csv("dataset/train.csv")

# Split to match the standard holdout check
# train_split: Day 48 + Day 49 slots 0-7 (up to row 77000)
# exam_split: Day 49 slot 8 (rows 77000 to end)
train_split = train.iloc[:77000].copy()
exam_split = train.iloc[77000:].copy()

train_split['geohash_raw'] = train_split['geohash'].astype(str)
exam_split['geohash_raw'] = exam_split['geohash'].astype(str)

# Preprocess time
def process_time(df):
    df['dt'] = pd.to_datetime(df['timestamp'], format='%H:%M')
    df['hour'] = df['dt'].dt.hour
    df['minute'] = df['dt'].dt.minute
    df['time_slot'] = df['hour'] * 4 + df['minute'] // 15
    df['time_slot_sin'] = np.sin(2 * np.pi * df['time_slot'] / 96)
    df['time_slot_cos'] = np.cos(2 * np.pi * df['time_slot'] / 96)
    df['day_of_week'] = df['day'] % 7
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_peak_hour'] = (((df['hour'] >= 7) & (df['hour'] <= 10)) | ((df['hour'] >= 17) & (df['hour'] <= 20))).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] < 6)).astype(int)
    df['is_office_hour'] = ((df['hour'] >= 9) & (df['hour'] < 17) & (df['is_weekend'] == 0)).astype(int)
    df.drop(columns=['dt'], inplace=True)
    return df

train_split = process_time(train_split)
exam_split = process_time(exam_split)

# Decode geohash to lat/lon
geohashes = pd.concat([train_split['geohash_raw'], exam_split['geohash_raw']]).unique()
geo_coords = {}
for g in geohashes:
    try:
        lat, lon = pgh.decode(g)
        geo_coords[g] = (lat, lon)
    except Exception:
        geo_coords[g] = (np.nan, np.nan)

train_split['lat'] = train_split['geohash_raw'].map(lambda g: geo_coords[g][0])
train_split['lon'] = train_split['geohash_raw'].map(lambda g: geo_coords[g][1])
exam_split['lat'] = exam_split['geohash_raw'].map(lambda g: geo_coords[g][0])
exam_split['lon'] = exam_split['geohash_raw'].map(lambda g: geo_coords[g][1])

train_split['lat'] = train_split['lat'].fillna(train_split['lat'].mean())
train_split['lon'] = train_split['lon'].fillna(train_split['lon'].mean())
exam_split['lat'] = exam_split['lat'].fillna(train_split['lat'].mean())
exam_split['lon'] = exam_split['lon'].fillna(train_split['lon'].mean())

# Spatial clustering
kmeans = KMeans(n_clusters=10, random_state=42, n_init='auto')
kmeans.fit(train_split[['lat', 'lon']])
train_split['spatial_cluster'] = kmeans.labels_
exam_split['spatial_cluster'] = kmeans.predict(exam_split[['lat', 'lon']])

# Collapse rare geohashes ONLY for categorical features, NOT for lookups
geohash_counts = train_split['geohash_raw'].value_counts()
rare_geohashes = set(geohash_counts[geohash_counts < 5].index)
train_split['geohash_cat'] = train_split['geohash_raw'].apply(lambda g: 'OTHER' if g in rare_geohashes else g)
exam_split['geohash_cat'] = exam_split['geohash_raw'].apply(lambda g: 'OTHER' if g in rare_geohashes else g)

# Impute temperature, weather, road type
temp_median = train_split['Temperature'].median()
geo_temp_median = train_split.groupby('geohash_raw')['Temperature'].median().to_dict()

def impute_temp(row):
    if pd.isna(row['Temperature']):
        return geo_temp_median.get(row['geohash_raw'], temp_median)
    return row['Temperature']

train_split['Temperature'] = train_split.apply(impute_temp, axis=1)
exam_split['Temperature'] = exam_split.apply(impute_temp, axis=1)

train_split['Weather'] = train_split['Weather'].fillna("Unknown")
exam_split['Weather'] = exam_split['Weather'].fillna("Unknown")
train_split['RoadType'] = train_split['RoadType'].fillna("Unknown")
exam_split['RoadType'] = exam_split['RoadType'].fillna("Unknown")

# Prefixes
train_split['geo_4'] = train_split['geohash_raw'].str[:4]
train_split['geo_5'] = train_split['geohash_raw'].str[:5]
exam_split['geo_4'] = exam_split['geohash_raw'].str[:4]
exam_split['geo_5'] = exam_split['geohash_raw'].str[:5]

# Interactions
def create_interactions(df):
    df['road_hour'] = df['RoadType'].astype(str) + "_" + df['hour'].astype(str)
    df['weather_hour'] = df['Weather'].astype(str) + "_" + df['hour'].astype(str)
    df['lane_hour'] = df['NumberofLanes'].astype(str) + "_" + df['hour'].astype(str)
    df['geo_hour'] = df['geohash_raw'] + "_" + df['hour'].astype(str)
    return df

train_split = create_interactions(train_split)
exam_split = create_interactions(exam_split)

# Lags and Anchors on Day 48 (Fully observed, no target leakage)
train_48 = train_split[train_split['day'] == 48].copy()
train_49 = train_split[train_split['day'] == 49].copy()

d48_demand_dict = train_48.set_index(['geohash_raw', 'time_slot'])['demand'].to_dict()

def get_d48_demand(geohash, slot, offset=0):
    t_slot = slot + offset
    if t_slot < 0 or t_slot > 95:
        return np.nan
    return d48_demand_dict.get((geohash, t_slot), np.nan)

# Add Day 48 demand lookups
for df in [train_49, exam_split]:
    df['demand_prev_day'] = df.apply(lambda r: get_d48_demand(r['geohash_raw'], r['time_slot'], 0), axis=1)
    df['lag_1_d48'] = df.apply(lambda r: get_d48_demand(r['geohash_raw'], r['time_slot'], -1), axis=1)
    df['lag_2_d48'] = df.apply(lambda r: get_d48_demand(r['geohash_raw'], r['time_slot'], -2), axis=1)
    df['lag_3_d48'] = df.apply(lambda r: get_d48_demand(r['geohash_raw'], r['time_slot'], -3), axis=1)
    
    slot_mean_d48 = train_48.groupby('time_slot')['demand'].mean().to_dict()
    df['slot_mean_d48'] = df['time_slot'].map(slot_mean_d48)
    
    for col in ['demand_prev_day', 'lag_1_d48', 'lag_2_d48', 'lag_3_d48']:
        df[col] = df[col].fillna(df['slot_mean_d48'])

# Statistical Aggregations on Day 48
geo_mean_d48 = train_48.groupby('geohash_raw')['demand'].mean().to_dict()
geo_std_d48 = train_48.groupby('geohash_raw')['demand'].std().to_dict()
geo_hour_mean_d48 = train_48.groupby('geo_hour')['demand'].mean().to_dict()
hour_mean_d48 = train_48.groupby('hour')['demand'].mean().to_dict()
weather_mean_d48 = train_48.groupby('Weather')['demand'].mean().to_dict()
slot_mean_d48 = train_48.groupby('time_slot')['demand'].mean().to_dict()

for df in [train_48, train_49, exam_split]:
    df['geo_mean_d48'] = df['geohash_raw'].map(geo_mean_d48).fillna(df['time_slot'].map(slot_mean_d48))
    df['geo_std_d48'] = df['geohash_raw'].map(geo_std_d48).fillna(0.0)
    df['geo_hour_mean_d48'] = df['geo_hour'].map(geo_hour_mean_d48).fillna(df['geo_mean_d48'])
    df['hour_mean_d48'] = df['hour'].map(hour_mean_d48).fillna(df['time_slot'].map(slot_mean_d48))
    df['weather_mean_d48'] = df['Weather'].map(weather_mean_d48).fillna(df['time_slot'].map(slot_mean_d48))
    df['slot_mean_d48'] = df['time_slot'].map(slot_mean_d48)

# Frequency Encodings
combined_train = pd.concat([train_48, train_49], ignore_index=True)
freq_cols = ['geohash_raw', 'geo_hour', 'road_hour']
for col in freq_cols:
    freq_map = combined_train[col].value_counts().to_dict()
    train_48[f'{col}_freq'] = train_48[col].map(freq_map)
    train_49[f'{col}_freq'] = train_49[col].map(freq_map)
    exam_split[f'{col}_freq'] = exam_split[col].map(freq_map).fillna(0.0)

# Categorical Encodings
cat_cols = ['Weather', 'RoadType', 'LargeVehicles', 'Landmarks', 'geohash_cat', 'geo_4', 'geo_5', 'road_hour', 'weather_hour', 'lane_hour', 'geo_hour']
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train_48[col].astype(str), train_49[col].astype(str), exam_split[col].astype(str)])
    le.fit(combined)
    train_48[col] = le.transform(train_48[col].astype(str))
    train_49[col] = le.transform(train_49[col].astype(str))
    exam_split[col] = le.transform(exam_split[col].astype(str))

# Base Features (No Day 48 lags, completely leak-free and time-robust!)
base_features = [
    'geohash_cat', 'geo_4', 'geo_5', 'time_slot_sin', 'time_slot_cos',
    'lat', 'lon', 'RoadType', 'NumberofLanes', 'LargeVehicles', 'Landmarks',
    'Temperature', 'Weather',
    'geo_mean_d48', 'geo_std_d48', 'geo_hour_mean_d48', 'hour_mean_d48', 'weather_mean_d48', 'slot_mean_d48',
    'road_hour', 'weather_hour', 'lane_hour', 'geo_hour',
    'geohash_raw_freq', 'geo_hour_freq', 'road_hour_freq', 'is_weekend', 'is_peak_hour', 'is_night', 'is_office_hour', 'spatial_cluster'
]

# Step 1: Train the base ensemble on combined Day 48 + Day 49 train_49
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_base_preds = np.zeros(len(train_49))
exam_base_preds = np.zeros(len(exam_split))

print("=== Training Base Spatial-Temporal Ensemble ===")
for fold, (train_idx, val_idx) in enumerate(kf.split(train_49)):
    train_49_fold = train_49.iloc[train_idx].copy()
    val_fold = train_49.iloc[val_idx].copy()
    
    combined_train_fold = pd.concat([train_48, train_49_fold], ignore_index=True)
    
    X_tr = combined_train_fold[base_features].copy()
    y_tr = combined_train_fold['demand'].copy()
    X_va = val_fold[base_features].copy()
    y_va = val_fold['demand'].copy()
    X_ex = exam_split[base_features].copy()
    
    # LightGBM
    lgb = LGBMRegressor(n_estimators=1000, learning_rate=0.03, num_leaves=63, random_state=42, verbose=-1)
    lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[early_stopping(stopping_rounds=50, verbose=False)])
    
    # XGBoost
    xgb = XGBRegressor(n_estimators=1000, learning_rate=0.03, max_depth=6, random_state=42, verbosity=0)
    xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    
    # CatBoost
    cat = CatBoostRegressor(iterations=1000, learning_rate=0.03, depth=6, random_state=42, verbose=0)
    cat.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    
    # Predict on validation and exam splits
    val_pred = (0.5 * lgb.predict(X_va) + 0.35 * xgb.predict(X_va) + 0.15 * cat.predict(X_va))
    ex_pred = (0.5 * lgb.predict(X_ex) + 0.35 * xgb.predict(X_ex) + 0.15 * cat.predict(X_ex))
    
    oof_base_preds[val_idx] = val_pred
    exam_base_preds += ex_pred / 5.0

print(f"Base Ensemble OOF R2 (slots 0-8): {r2_score(train_49['demand'], oof_base_preds):.5f}")

# Step 2: Fit a Linear Calibration Model on Day 49 slots 0-8
# We learn: demand_49 = w1 * base_pred + w2 * demand_prev_day + bias
X_cal = pd.DataFrame({
    'base_pred': oof_base_preds,
    'demand_prev_day': train_49['demand_prev_day'].values
})
y_cal = train_49['demand'].values

calibrator = Ridge(alpha=1.0)
calibrator.fit(X_cal, y_cal)

print("\n--- Calibration Model Coefficients ---")
print(f"Base Pred Weight:      {calibrator.coef_[0]:.5f}")
print(f"Demand Prev Day Weight: {calibrator.coef_[1]:.5f}")
print(f"Bias:                  {calibrator.intercept_:.5f}")

# Step 3: Calibrate Holdout Exam Split predictions
X_ex_cal = pd.DataFrame({
    'base_pred': exam_base_preds,
    'demand_prev_day': exam_split['demand_prev_day'].values
})

final_exam_preds = np.clip(calibrator.predict(X_ex_cal), 0.0, 1.0)
r2_final = r2_score(exam_split['demand'], final_exam_preds)

print("\n========================================================")
print("             CALIBRATION STACK EXAM RESULTS             ")
print("========================================================")
print(f"Final Calibration Stack Holdout R2: {r2_final:.5f} ({r2_final*100:.2f}%)")
print("========================================================")
