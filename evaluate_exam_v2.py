import pandas as pd
import numpy as np
import pygeohash as pgh
from lightgbm import LGBMRegressor, early_stopping
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("=== Starting V2 Holdout Exam Evaluation (Same-Day Lags Pipeline) ===")

# 1. Load data
print("Loading train.csv...")
train = pd.read_csv("dataset/train.csv")

# Split into train (first 77000 rows) and exam (remaining 299 rows)
train_split = train.iloc[:77000].copy()
exam_split = train.iloc[77000:].copy()

print(f"Train split size: {len(train_split)}")
print(f"Exam split size: {len(exam_split)}")

train_split['geohash_raw'] = train_split['geohash'].astype(str)
exam_split['geohash_raw'] = exam_split['geohash'].astype(str)

# 2. Preprocess time
def process_time(df):
    df['dt'] = pd.to_datetime(df['timestamp'], format='%H:%M')
    df['hour'] = df['dt'].dt.hour
    df['minute'] = df['dt'].dt.minute
    df['time_slot'] = df['hour'] * 4 + df['minute'] // 15
    df['time_slot_sin'] = np.sin(2 * np.pi * df['time_slot'] / 96)
    df['time_slot_cos'] = np.cos(2 * np.pi * df['time_slot'] / 96)
    df['day_of_week'] = df['day'] % 7
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Peak hour, night, and office hour indicators
    df['is_peak_hour'] = (((df['hour'] >= 7) & (df['hour'] <= 10)) | ((df['hour'] >= 17) & (df['hour'] <= 20))).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] < 6)).astype(int)
    df['is_office_hour'] = ((df['hour'] >= 9) & (df['hour'] < 17) & (df['is_weekend'] == 0)).astype(int)
    
    df.drop(columns=['dt'], inplace=True)
    return df

train_split = process_time(train_split)
exam_split = process_time(exam_split)

# 3. Decode geohash to lat/lon
print("Decoding geohashes to lat/lon...")
geohashes = pd.concat([train_split['geohash_raw'], exam_split['geohash_raw']]).unique()
geo_coords = {g: pgh.decode(g) for g in geohashes}
train_split['lat'] = train_split['geohash_raw'].map(lambda g: geo_coords[g][0])
train_split['lon'] = train_split['geohash_raw'].map(lambda g: geo_coords[g][1])
exam_split['lat'] = exam_split['geohash_raw'].map(lambda g: geo_coords[g][0])
exam_split['lon'] = exam_split['geohash_raw'].map(lambda g: geo_coords[g][1])

train_split['lat'] = train_split['lat'].fillna(train_split['lat'].mean())
train_split['lon'] = train_split['lon'].fillna(train_split['lon'].mean())
exam_split['lat'] = exam_split['lat'].fillna(train_split['lat'].mean())
exam_split['lon'] = exam_split['lon'].fillna(train_split['lon'].mean())

# 4. Spatial Clustering (KMeans on lat/lon)
print("Fitting spatial clustering...")
kmeans = KMeans(n_clusters=10, random_state=42, n_init='auto')
kmeans.fit(train_split[['lat', 'lon']])
train_split['spatial_cluster'] = kmeans.labels_
exam_split['spatial_cluster'] = kmeans.predict(exam_split[['lat', 'lon']])

train_48 = train_split[train_split['day'] == 48].copy()
train_49 = train_split[train_split['day'] == 49].copy()

# Lookups
d48_demand_dict = train_48.set_index(['geohash_raw', 'time_slot'])['demand'].to_dict()
d49_demand_dict = train_49.set_index(['geohash_raw', 'time_slot'])['demand'].to_dict()

def get_demand(day, geohash, slot, offset=0):
    t_slot = slot + offset
    if t_slot < 0 or t_slot > 95:
        return np.nan
    if day == 48:
        return d48_demand_dict.get((geohash, t_slot), np.nan)
    elif day == 49:
        return d49_demand_dict.get((geohash, t_slot), np.nan)
    return np.nan

# Group stats on Day 48
geo_mean_d48 = train_48.groupby('geohash_raw')['demand'].mean().to_dict()
geo_std_d48 = train_48.groupby('geohash_raw')['demand'].std().to_dict()
slot_mean_d48 = train_48.groupby('time_slot')['demand'].mean().to_dict()

# 5. Interaction keys
print("Engineering interaction features...")
def create_interactions(df):
    df['road_hour'] = df['RoadType'].astype(str) + "_" + df['hour'].astype(str)
    df['weather_hour'] = df['Weather'].astype(str) + "_" + df['hour'].astype(str)
    df['lane_hour'] = df['NumberofLanes'].astype(str) + "_" + df['hour'].astype(str)
    df['geo_hour'] = df['geohash_raw'] + "_" + df['hour'].astype(str)
    return df

train_split = create_interactions(train_split)
exam_split = create_interactions(exam_split)

train_48 = train_split[train_split['day'] == 48].copy()
train_49 = train_split[train_split['day'] == 49].copy()

geo_hour_mean_d48 = train_48.groupby('geo_hour')['demand'].mean().to_dict()
hour_mean_d48 = train_48.groupby('hour')['demand'].mean().to_dict()
weather_mean_d48 = train_48.groupby('Weather')['demand'].mean().to_dict()

# 6. Set up lags (Same-day lags only to preserve consistency)
print("Constructing same-day lags & group statistics...")
for df in [train_48, train_49, exam_split]:
    day_val = df['day'].iloc[0]
    df['same_day_lag_1'] = df.apply(lambda r: get_demand(day_val, r['geohash_raw'], r['time_slot'], -1), axis=1)
    df['same_day_lag_2'] = df.apply(lambda r: get_demand(day_val, r['geohash_raw'], r['time_slot'], -2), axis=1)
    df['same_day_lag_3'] = df.apply(lambda r: get_demand(day_val, r['geohash_raw'], r['time_slot'], -3), axis=1)

# Impute lags & rolling stats
for df in [train_48, train_49, exam_split]:
    slot_mean_val = df['time_slot'].map(slot_mean_d48)
    
    df['same_day_lag_1'] = df['same_day_lag_1'].fillna(slot_mean_val)
    df['same_day_lag_2'] = df['same_day_lag_2'].fillna(slot_mean_val)
    df['same_day_lag_3'] = df['same_day_lag_3'].fillna(slot_mean_val)
        
    df['geo_mean_d48'] = df['geohash_raw'].map(geo_mean_d48).fillna(slot_mean_val)
    df['geo_std_d48'] = df['geohash_raw'].map(geo_std_d48).fillna(0.0)
    df['geo_hour_mean_d48'] = df['geo_hour'].map(geo_hour_mean_d48).fillna(df['geo_mean_d48'])
    df['hour_mean_d48'] = df['hour'].map(hour_mean_d48).fillna(slot_mean_val)
    df['weather_mean_d48'] = df['Weather'].map(weather_mean_d48).fillna(slot_mean_val)
    df['slot_mean_d48'] = slot_mean_val
    
    # Advanced Same-day features
    df['rolling_mean_3_same'] = (df['same_day_lag_1'] + df['same_day_lag_2'] + df['same_day_lag_3']) / 3.0
    df['diff_1_same'] = df['same_day_lag_1'] - df['same_day_lag_2']
    
    df['geo_4'] = df['geohash_raw'].str[:4]
    df['geo_5'] = df['geohash_raw'].str[:5]

# 7. Frequency Encodings
print("Computing frequency encodings...")
combined_train = pd.concat([train_48, train_49], ignore_index=True)
freq_cols = ['geohash_raw', 'geo_hour', 'road_hour']
for col in freq_cols:
    freq_map = combined_train[col].value_counts().to_dict()
    train_48[f'{col}_freq'] = train_48[col].map(freq_map)
    train_49[f'{col}_freq'] = train_49[col].map(freq_map)
    exam_split[f'{col}_freq'] = exam_split[col].map(freq_map).fillna(0.0)

# 8. Categorical Encodings
print("Encoding categorical features...")
cat_cols = ['Weather', 'RoadType', 'LargeVehicles', 'Landmarks', 'geo_4', 'geo_5', 'road_hour', 'weather_hour', 'lane_hour']
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train_48[col].astype(str), train_49[col].astype(str), exam_split[col].astype(str)])
    le.fit(combined)
    train_48[col] = le.transform(train_48[col].astype(str))
    train_49[col] = le.transform(train_49[col].astype(str))
    exam_split[col] = le.transform(exam_split[col].astype(str))

# Define feature space
features = [
    'geo_4', 'geo_5', 'time_slot_sin', 'time_slot_cos',
    'lat', 'lon', 'RoadType', 'NumberofLanes', 'LargeVehicles', 'Landmarks',
    'Temperature', 'Weather', 'same_day_lag_1', 'same_day_lag_2', 'same_day_lag_3',
    'rolling_mean_3_same', 'diff_1_same',
    'geo_mean_d48', 'geo_std_d48', 'geo_hour_mean_d48', 'hour_mean_d48', 'weather_mean_d48', 'slot_mean_d48',
    'road_hour', 'weather_hour', 'lane_hour',
    'geohash_raw_freq', 'geo_hour_freq', 'road_hour_freq',
    'is_weekend', 'is_peak_hour', 'is_night', 'is_office_hour', 'spatial_cluster'
]

# 9. Fold-Safe CV on Day 49
kf = KFold(n_splits=5, shuffle=True, random_state=42)
preds_lgb = np.zeros(len(exam_split))
preds_xgb = np.zeros(len(exam_split))
preds_cat = np.zeros(len(exam_split))

print("\n=== Training models on combined Day 48 & Day 49 splits ===")
for fold, (train_idx, val_idx) in enumerate(kf.split(train_49)):
    print(f"--- FOLD {fold+1} ---")
    train_49_fold = train_49.iloc[train_idx].copy()
    val_fold = train_49.iloc[val_idx].copy()
    
    combined_train_fold = pd.concat([train_48, train_49_fold], ignore_index=True)
    
    X_tr = combined_train_fold[features].copy()
    y_tr = combined_train_fold['demand'].copy()
    X_va = val_fold[features].copy()
    y_va = val_fold['demand'].copy()
    X_ex = exam_split[features].copy()
    
    # LGBM
    lgb = LGBMRegressor(n_estimators=2000, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1)
    lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[early_stopping(50, verbose=False)])
    preds_lgb += lgb.predict(X_ex) / 5.0
    
    # XGBoost
    xgb = XGBRegressor(n_estimators=2000, learning_rate=0.03, max_depth=5, early_stopping_rounds=50, random_state=42, verbosity=0)
    xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    preds_xgb += xgb.predict(X_ex) / 5.0
    
    # CatBoost
    cat = CatBoostRegressor(iterations=2000, learning_rate=0.03, depth=5, early_stopping_rounds=50, random_state=42, verbose=0)
    cat.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    preds_cat += cat.predict(X_ex) / 5.0

# 10. Blend predictions
w_lgb, w_xgb, w_cat = 0.05, 0.14, 0.81
exam_preds = w_lgb * preds_lgb + w_xgb * preds_xgb + w_cat * preds_cat
exam_preds_clipped = np.clip(exam_preds, 0.0, 1.0)

# Calculate final metrics
y_exam = exam_split['demand']
r2 = r2_score(y_exam, exam_preds_clipped)
mae = mean_absolute_error(y_exam, exam_preds_clipped)
rmse = np.sqrt(mean_squared_error(y_exam, exam_preds_clipped))

print("\n=== Holdout Exam Performance Results (V2) ===")
print(f"Holdout R2 (Accuracy): {r2:.5f} ({r2*100:.2f}%)")
print(f"Holdout MAE:           {mae:.5f}")
print(f"Holdout RMSE:          {rmse:.5f}")
