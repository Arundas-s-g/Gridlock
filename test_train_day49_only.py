import pandas as pd
import numpy as np
import pygeohash as pgh
from lightgbm import LGBMRegressor, early_stopping
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_csv("dataset/train.csv")

# Split
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

# Collapse rare geohashes
geohash_counts = train_split['geohash_raw'].value_counts()
rare_geohashes = set(geohash_counts[geohash_counts < 5].index)
train_split['geohash_raw'] = train_split['geohash_raw'].apply(lambda g: 'OTHER' if g in rare_geohashes else g)
exam_split['geohash_raw'] = exam_split['geohash_raw'].apply(lambda g: 'OTHER' if g in rare_geohashes else g)

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

# Day 48 Lags & Stats
train_48 = train_split[train_split['day'] == 48].copy()
train_49 = train_split[train_split['day'] == 49].copy()

d48_demand_dict = train_48.set_index(['geohash_raw', 'time_slot'])['demand'].to_dict()

def get_d48_demand(geohash, slot, offset=0):
    t_slot = slot + offset
    if t_slot < 0 or t_slot > 95:
        return np.nan
    return d48_demand_dict.get((geohash, t_slot), np.nan)

for df in [train_49, exam_split]:
    df['demand_prev_day'] = df.apply(lambda r: get_d48_demand(r['geohash_raw'], r['time_slot'], 0), axis=1)
    df['lag_1_d48'] = df.apply(lambda r: get_d48_demand(r['geohash_raw'], r['time_slot'], -1), axis=1)
    df['lag_2_d48'] = df.apply(lambda r: get_d48_demand(r['geohash_raw'], r['time_slot'], -2), axis=1)
    df['lag_3_d48'] = df.apply(lambda r: get_d48_demand(r['geohash_raw'], r['time_slot'], -3), axis=1)
    df['lag_6_d48'] = df.apply(lambda r: get_d48_demand(r['geohash_raw'], r['time_slot'], -6), axis=1)
    df['lag_12_d48'] = df.apply(lambda r: get_d48_demand(r['geohash_raw'], r['time_slot'], -12), axis=1)
    df['lag_24_d48'] = df.apply(lambda r: get_d48_demand(r['geohash_raw'], r['time_slot'], -24), axis=1)
    
    slot_mean_d48 = train_48.groupby('time_slot')['demand'].mean().to_dict()
    df['slot_mean_d48'] = df['time_slot'].map(slot_mean_d48)
    
    for col in ['demand_prev_day', 'lag_1_d48', 'lag_2_d48', 'lag_3_d48', 'lag_6_d48', 'lag_12_d48', 'lag_24_d48']:
        df[col] = df[col].fillna(df['slot_mean_d48'])
        
    df['rolling_mean_3_d48'] = (df['lag_1_d48'] + df['demand_prev_day'] + df.apply(lambda r: get_d48_demand(r['geohash_raw'], r['time_slot'], 1), axis=1).fillna(df['slot_mean_d48'])) / 3.0
    df['rolling_mean_6_d48'] = (df['lag_2_d48'] + df['lag_1_d48'] + df['demand_prev_day'] + 
                                df.apply(lambda r: get_d48_demand(r['geohash_raw'], r['time_slot'], 1), axis=1).fillna(df['slot_mean_d48']) + 
                                df.apply(lambda r: get_d48_demand(r['geohash_raw'], r['time_slot'], 2), axis=1).fillna(df['slot_mean_d48']) + 
                                df.apply(lambda r: get_d48_demand(r['geohash_raw'], r['time_slot'], 3), axis=1).fillna(df['slot_mean_d48'])) / 6.0
    df['rolling_mean_12_d48'] = (df['lag_6_d48'] + df['lag_3_d48'] + df['lag_2_d48'] + df['lag_1_d48'] + df['demand_prev_day'] + 
                                 df.apply(lambda r: get_d48_demand(r['geohash_raw'], r['time_slot'], 1), axis=1).fillna(df['slot_mean_d48']) + 
                                 df.apply(lambda r: get_d48_demand(r['geohash_raw'], r['time_slot'], 2), axis=1).fillna(df['slot_mean_d48']) + 
                                 df.apply(lambda r: get_d48_demand(r['geohash_raw'], r['time_slot'], 3), axis=1).fillna(df['slot_mean_d48']) +
                                 df.apply(lambda r: get_d48_demand(r['geohash_raw'], r['time_slot'], 4), axis=1).fillna(df['slot_mean_d48']) +
                                 df.apply(lambda r: get_d48_demand(r['geohash_raw'], r['time_slot'], 5), axis=1).fillna(df['slot_mean_d48']) +
                                 df.apply(lambda r: get_d48_demand(r['geohash_raw'], r['time_slot'], 6), axis=1).fillna(df['slot_mean_d48'])) / 11.0
                                 
    df['rolling_std_3_d48'] = np.std([df['lag_1_d48'], df['demand_prev_day'], df.apply(lambda r: get_d48_demand(r['geohash_raw'], r['time_slot'], 1), axis=1).fillna(df['slot_mean_d48'])], axis=0)
    df['rolling_max_3_d48'] = np.max([df['lag_1_d48'], df['demand_prev_day'], df.apply(lambda r: get_d48_demand(r['geohash_raw'], r['time_slot'], 1), axis=1).fillna(df['slot_mean_d48'])], axis=0)
    df['rolling_min_3_d48'] = np.min([df['lag_1_d48'], df['demand_prev_day'], df.apply(lambda r: get_d48_demand(r['geohash_raw'], r['time_slot'], 1), axis=1).fillna(df['slot_mean_d48'])], axis=0)

# Statistical Aggregations on Day 48
geo_mean_d48 = train_48.groupby('geohash_raw')['demand'].mean().to_dict()
geo_std_d48 = train_48.groupby('geohash_raw')['demand'].std().to_dict()
geo_hour_mean_d48 = train_48.groupby('geo_hour')['demand'].mean().to_dict()
hour_mean_d48 = train_48.groupby('hour')['demand'].mean().to_dict()
weather_mean_d48 = train_48.groupby('Weather')['demand'].mean().to_dict()
slot_mean_d48 = train_48.groupby('time_slot')['demand'].mean().to_dict()

for df in [train_49, exam_split]:
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
    train_49[f'{col}_freq'] = train_49[col].map(freq_map)
    exam_split[f'{col}_freq'] = exam_split[col].map(freq_map).fillna(0.0)

# Categorical Encodings
cat_cols = ['Weather', 'RoadType', 'LargeVehicles', 'Landmarks', 'geohash', 'geo_4', 'geo_5', 'road_hour', 'weather_hour', 'lane_hour', 'geo_hour']
for col in cat_cols:
    le = LabelEncoder()
    if col == 'geohash':
        combined = pd.concat([train_49['geohash_raw'].astype(str), exam_split['geohash_raw'].astype(str)])
        le.fit(combined)
        train_49['geohash_encoded'] = le.transform(train_49['geohash_raw'].astype(str))
        exam_split['geohash_encoded'] = le.transform(exam_split['geohash_raw'].astype(str))
    else:
        combined = pd.concat([train_49[col].astype(str), exam_split[col].astype(str)])
        le.fit(combined)
        train_49[col] = le.transform(train_49[col].astype(str))
        exam_split[col] = le.transform(exam_split[col].astype(str))

features = [
    'geohash_encoded', 'geo_4', 'geo_5', 'time_slot_sin', 'time_slot_cos',
    'lat', 'lon', 'RoadType', 'NumberofLanes', 'LargeVehicles', 'Landmarks',
    'Temperature', 'Weather', 'demand_prev_day',
    'lag_1_d48', 'lag_2_d48', 'lag_3_d48', 'lag_6_d48', 'lag_12_d48', 'lag_24_d48',
    'rolling_mean_3_d48', 'rolling_mean_6_d48', 'rolling_mean_12_d48', 'rolling_std_3_d48', 'rolling_max_3_d48', 'rolling_min_3_d48',
    'geo_mean_d48', 'geo_std_d48', 'geo_hour_mean_d48', 'hour_mean_d48', 'weather_mean_d48', 'slot_mean_d48',
    'road_hour', 'weather_hour', 'lane_hour', 'geo_hour',
    'geohash_raw_freq', 'geo_hour_freq', 'road_hour_freq', 'is_weekend', 'is_peak_hour', 'is_night', 'is_office_hour', 'spatial_cluster'
]

# Fold-Safe Target Encoding CV over Train day 49
kf = KFold(n_splits=5, shuffle=True, random_state=42)
exam_preds = np.zeros(len(exam_split))
te_cols = ['geohash_encoded', 'geo_hour', 'road_hour']

for fold, (train_idx, val_idx) in enumerate(kf.split(train_49)):
    train_49_fold = train_49.iloc[train_idx].copy()
    val_fold = train_49.iloc[val_idx].copy()
    
    X_tr = train_49_fold[features].copy()
    y_tr = train_49_fold['demand'].copy()
    X_va = val_fold[features].copy()
    y_va = val_fold['demand'].copy()
    X_ex = exam_split[features].copy()
    
    # Fold-safe target encoding
    for col in te_cols:
        target_mean = y_tr.groupby(X_tr[col]).mean().to_dict()
        global_mean = y_tr.mean()
        X_tr[f'{col}_te'] = X_tr[col].map(target_mean).fillna(global_mean)
        X_va[f'{col}_te'] = X_va[col].map(target_mean).fillna(global_mean)
        X_ex[f'{col}_te'] = X_ex[col].map(target_mean).fillna(global_mean)
        
    X_tr_fit = X_tr.drop(columns=te_cols)
    X_va_fit = X_va.drop(columns=te_cols)
    X_ex_fit = X_ex.drop(columns=te_cols)
    
    # LGBM
    lgb = LGBMRegressor(n_estimators=3000, learning_rate=0.03, num_leaves=63, min_child_samples=50, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l1=0.1, lambda_l2=0.1, random_state=42, verbose=-1)
    lgb.fit(X_tr_fit, y_tr, eval_set=[(X_va_fit, y_va)], callbacks=[early_stopping(stopping_rounds=100, verbose=False)])
    pred_ex_lgb = lgb.predict(X_ex_fit)
    
    # XGBoost
    xgb = XGBRegressor(n_estimators=3000, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=100, random_state=42, verbosity=0)
    xgb.fit(X_tr_fit, y_tr, eval_set=[(X_va_fit, y_va)], verbose=False)
    pred_ex_xgb = xgb.predict(X_ex_fit)
    
    # CatBoost
    cat = CatBoostRegressor(iterations=3000, learning_rate=0.03, depth=6, subsample=0.8, early_stopping_rounds=100, random_state=42, verbose=0)
    cat.fit(X_tr_fit, y_tr, eval_set=[(X_va_fit, y_va)], verbose=False)
    pred_ex_cat = cat.predict(X_ex_fit)
    
    exam_preds += (0.5 * pred_ex_lgb + 0.35 * pred_ex_xgb + 0.15 * pred_ex_cat) / 5.0

exam_preds_clipped = np.clip(exam_preds, 0.0, 1.0)
y_exam = exam_split['demand']
r2 = r2_score(y_exam, exam_preds_clipped)
print(f"DAY 49 ONLY TRAINING - HOLD OUT EXAM R2: {r2:.5f} ({r2*100:.2f}%)")
